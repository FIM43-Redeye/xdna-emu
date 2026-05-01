---
date: 2026-05-01
status: phases-1+2-landed, phase-3-pending
relates_to: bridge-test #318, mode-2 trace divergence
sources: AM020 §2 (chapter-2-aie-ml-tile-architecture.md L299-305), live HW
  trace from add_one_using_dma.chess pc-anchored sweep, Chess listing
  main_core_0_2.elf.lst
---

# Mode-2 (INST_EXEC) trace: atom and New_PC semantics

## TL;DR

Our emulator emits mode-2 frames with the wrong semantics. AM020 specifies
mode 2 as a **branch-only** trace ("minimum set of information to allow an
offline debugger to reconstruct the program execution flow"), but our
emitter writes one E/N atom **per cycle**. On `add_one_using_dma`:

| Quantity   | HW  | EMU | Ratio |
|------------|-----|-----|-------|
| E_atom     | 4   | 582 | 145x  |
| N_atom     | 2   | 157 | 78x   |
| New_PC     | 34  | 69  | 2x    |
| anchor_pc  | 832 | 0   | (n/a) |

## What HW actually records

AM020 §2 (chapter-2, L299-305) -- direct quote:

> The trace unit in the AIE-ML can operate in execution-trace mode. In real
> time, the unit will send, via the AXI4-Stream, a minimum set of
> information to allow an offline debugger to reconstruct the program
> execution flow. This assumes the offline debugger has access to the ELF.
> The information includes:
> - Conditional and unconditional direct branches
> - All indirect branches
> - Zero-overhead-loop LC

That is exactly an ARM-ETM-style branch trace. Atoms and New_PCs are emitted
**only at branch resolution**, never per cycle.

## The HW pattern, decoded

Captured trace for `core_0_2` (chess):

```
Start anchor_pc=832
[36 frames, broken into 4 iterations of the body loop]
New_PC 368, 496, 528, 560, 592, 704, 736, E_atom         iter 1 (anchor inside acquire)
New_PC 336, 368, 496, 528, 560, 592, 704, 736, E_atom    iter 2
New_PC 336, 368, 496, 528, 560, 592, 704, 736, E_atom    iter 3
New_PC 336, 368, 496, 528, 560, 592, 704, 736, N_atom    iter 4 (loop exits)
New_PC 144, 880, E_atom                                   post-loop continuation
New_PC 1184, N_atom
Sync
```

Mapping the iter-2 PCs to the chess listing:

| PC  | Origin                              | Why traced                                |
|-----|-------------------------------------|-------------------------------------------|
| 336 | `JL #816` (call acquire)            | JNZD destination from prior iter (loop top) |
| 368 | `LDA r25, [p7], #4` (after JL+5DS)  | RET destination from acquire (dynamic)    |
| 496 | `JL #848` (call release)            | RET destination from prior call at 436    |
| 528 | `JL #816` (call acquire)            | RET destination                           |
| 560 | `JL #816` (call acquire)            | RET destination                           |
| 592 | `LDA r25, [p6], #4` (after JL+5DS)  | RET destination                           |
| 704 | `JL #848` (call release)            | RET destination from prior call at 660    |
| 736 | `JNZD r17, r17, p7` (loop-back)     | New_PC at conditional-indirect branch     |

**Anchor 832** is `NOPA;NOPB;NOPS;NOPX;NOPM;NOPV` inside
`llvm___aie2___acquire` (the lock-acquire spin/stall pad). The trace
captured the PC at the moment BROADCAST_15 (start_event=122) fired -- by
which time the core had already entered acquire and was stalling.

## Decoded rules

### When New_PC fires

**Only at dynamic-destination branches** -- where the debugger cannot deduce
the next PC from the ELF alone. Concretely:

- **RET** (`ret`): emit New_PC = caller's post-delay-slot resume PC (read
  from the link register at retire time). This dominates the trace because
  every JL+RET pair contributes one New_PC.
- **Indirect taken branches** (`JNZD r, r, p7`, indirect `J`/`JL` via
  register): emit New_PC = the runtime-resolved target.

**Direct unconditional branches do NOT emit New_PC** -- the target is
encoded in the JL/J/JC instruction itself; the debugger reads it from the
ELF. This is the source of our 2x New_PC inflation: we emit on every taken
branch, including direct JLs.

### When E_atom / N_atom fires

**Only at conditional branch resolution.** Each conditional branch
(`JZ`/`JNZ`/`JZD`/`JNZD`) emits exactly one atom:

- **E_atom** = condition fired, branch taken
- **N_atom** = condition false, fall-through

Unconditional branches and RETs emit no atoms (they always "execute" -- no
disposition to record).

The 4-3-1 E/N pattern in the iter loop is the JNZD outcome on each pass:
loop-counter > 0 three times (E, E, E), then = 0 (N).

### When LC fires

Once per ZOL invocation (already correct in our emitter, per
`2026-04-30-mode2-lc-flag-semantics.md`).

### What anchor_pc captures

**The core's PC at the cycle the start_event fires.** In HW, start_event is
BROADCAST_15 fired by the host write to Event_Generate; in `add_one_using_dma`
this happens after the kernel has begun and stalled in acquire, hence
anchor=832.

In EMU we currently fire start_event on cycle-0 TRUE (the metronome). PC at
cycle 0 is 0, so we record anchor_pc=0. This isn't a per-cycle plumbing bug
-- it's a start-timing mismatch.

## What the EMU does wrong

| File                                              | What it does                                  | What HW does                  |
|---------------------------------------------------|-----------------------------------------------|-------------------------------|
| `coordinator.rs:1102-1112`                        | Calls `notify_core_active/stalled` per cycle  | Atoms only at conditional br  |
| `cycle_accurate.rs:582-587`                       | Emits New_PC for every Branch or Call target  | Only RET / indirect taken     |
| `coordinator.rs:992-1008`                         | Fires TRUE+PC on every cycle to trace_unit    | start_event = BROADCAST_15    |

`trace_unit/mod.rs` itself is fine -- the encoding (push_bits, RLE,
align_to_word_via_filler0, encode_*) is correct. The bug is in **what the
interpreter reports to the trace_unit**.

## Plan to fix

### Phase 1 -- atom semantics (the big one)

1. **Remove per-cycle calls** to `notify_core_active`/`notify_core_stalled`
   in `coordinator.rs:1102-1112`. Drop or repurpose those two trace_unit
   methods.
2. **Add conditional-branch hook** in cycle_accurate executor: when the
   instruction is a conditional branch (`JZ`/`JNZ`/`JZD`/`JNZD`/`JEQZ`/etc.),
   emit `notify_atom(taken: bool)`. The trace_unit appends to the atom RLE
   run, polarity-flipping where needed.
3. The existing `pending_atoms_run` / `flush_atoms_run` /
   `encode_repeat0|1` machinery continues to work -- runs of E or N still
   compress correctly, just at branch granularity.

### Phase 2 -- New_PC semantics

1. **Stop emitting New_PC for direct unconditional branches.** In
   `cycle_accurate.rs:582-587`, check the instruction type: only emit
   `notify_branch_taken` when the destination is dynamic (RET, indirect
   J/JL, indirect conditional taken).
2. Need a way to classify the instruction. Two options:
   - extend `ExecuteResult` with a `Branch { target, dynamic: bool }`
     variant (or a separate `Ret` / `IndirectBranch` variant)
   - look up the SemanticOp from the bundle being executed and match on
     `Ret | IndirectJump | IndirectCall`
3. Conditional-taken with direct target: atom only, no New_PC. Conditional
   not-taken: atom only.

### Phase 3 -- anchor_pc / start_event

1. The metronome firing of TRUE+PC is unrelated to mode-2 start. mode-2's
   start_event in real configs is BROADCAST_15 (event 122). The
   broadcast/event subsystem already routes that.
2. We need to verify our XRT plugin path actually broadcasts BROADCAST_15
   at the same point the kernel issues it on real HW. Likely candidates for
   timing drift:
   - Plugin fires the broadcast immediately on submit_cmd (cycle 0)
   - HW fires it after the kernel context-load sequence completes
3. Worst case: if start-event timing is genuinely synchronous-with-cycle-0
   and the difference is only that HW's kernel happens to be in acquire
   stall by then, the fix may be "wait for first lock-acquire stall before
   firing the broadcast" or similar. Needs deeper trace of what the actual
   `aie_arch.mlir` runtime does on the host side.

### Out of scope for this fix

- Compaction / Repeat0 / Repeat1 emission: already correct, will simply
  appear less because there are fewer atoms.
- LC frame semantics: confirmed correct.
- mode-2 decoder: already correct (verified by decoding both HW and EMU
  binaries).

## Validation plan

After Phase 1 + 2:
- Re-run `./scripts/emu-bridge-test.sh -v add_one_using_dma --chess-only`
- Expect the mode-2 report to show ~6 atoms and ~34 New_PCs on EMU side
  (matching HW), with PC-sequence equality (or near-equality, modulo
  whatever cycle-level mismatch remains).
- After Phase 3: anchor_pc should be 832 instead of 0, but only if the
  broadcast timing is corrected.

After all three phases pass, generalize to the full bridge sweep (#305).

## Phase 1 + 2 results (2026-05-01)

Landed in:
- `src/device/trace_unit/mod.rs` -- replaced `notify_core_active`/
  `notify_core_stalled` with `notify_atom(taken: bool)`, doc-anchored to
  AM020.
- `src/interpreter/engine/coordinator.rs` -- removed Phase 3f's per-cycle
  atom firing.
- `src/interpreter/execute/control.rs` -- conditional branches call
  `notify_atom`; Ret + indirect Br/Call/BrCond call `notify_branch_taken`;
  direct unconditional and direct-conditional-not-taken emit nothing.
- `src/interpreter/execute/cycle_accurate.rs` -- removed the post-hoc
  `notify_branch_taken` call (control.rs owns it now, with semantic
  context intact).
- `src/device/trace_unit/tests.rs` -- updated 4 test sites to the new
  API; all 56 trace_unit tests pass; full lib suite 2852 / 2852.

Bridge-test results (`add_one_using_dma.chess`, mode-2 baseline):

| Quantity | HW  | EMU before | EMU after |
|----------|-----|------------|-----------|
| Total frames | 42 | 810       | **43**    |
| New_PC   | 34  | 69         | **37**    |
| E_atom   | 4   | 582        | **3**     |
| N_atom   | 2   | 157        | **1**     |
| Start    | 1   | 1          | 1         |
| Sync     | 1   | 1          | 1         |

The mode-2 comparator's PC-sequence diff went from `34/69` to `34/37`.
Frame structure now mirrors HW iteration-by-iteration.

### Remaining structural deltas (small, well-localised)

1. **Iter 1 leading 336 in EMU but not HW** (+1 New_PC): HW's anchor caught
   the trace mid-acquire-stall (anchor_pc=832), so the very first RET
   destination 336 was already past by the time tracing began. EMU's
   anchor_pc=0 means the trace records that first RET. Phase 3 fixes this.
2. **Iters 2-4 leading 304 in EMU but not HW** (+3 New_PC): the JNZD at
   PC 736 is conditional-indirect (`JNZD r17, r17, p7`). Per AM020 it
   should emit New_PC under "All indirect branches", but HW skips it.
   Hypothesis: HW may treat conditional-indirect destinations specially,
   or the runtime p7 differs between the two -- needs follow-up. Not
   blocking the comparator usefulness.
3. **Post-loop missing E_atom / N_atom / New_PC 1184 in EMU** (-3 frames):
   the kernel halts in EMU before reaching `_fini` 's branches that HW
   captures. This is a halt-detection question, not a trace-emission
   one. File as a separate follow-up if it persists across other tests.

Net: PC-sequence delta of 3 frames, atom-count delta of 2. Both are
narrow, localized issues -- no longer the sweeping per-cycle inflation.

## Cross-references

- `AM020 chapter-2-aie-ml-tile-architecture.md:299-305` -- spec quote
- `mlir-aie/build/test/npu-xrt/add_one_using_dma/traced/aie_traced.mlir.prj/main_core_0_2.elf.lst` -- chess listing
- `xdna-emu/src/device/trace_unit/mod.rs:570-595` -- current notify API
- `xdna-emu/src/interpreter/execute/cycle_accurate.rs:582-587` -- current branch hook
- `xdna-emu/src/interpreter/engine/coordinator.rs:1100-1116` -- current per-cycle atom firing
- `xdna-emu/docs/superpowers/findings/2026-04-30-mode2-lc-flag-semantics.md` -- LC findings
