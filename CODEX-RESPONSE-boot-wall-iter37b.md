# Codex adversarial response: the 0x2450 boot wall (iter37b)

**Date:** 2026-07-10  
**Input brief:** `CODEX-BRIEF-boot-wall-iter37b.md`  
**Firmware audited:** `../xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin`  
**Firmware SHA-256:** `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`

## Executive verdict

Do **not** implement the iter37b proposal to pre-commit `current` to `0x10dfc`
before trapping into `FUN_00002730`.

The low-level observations in the brief are mostly reproducible, including the
numeric call target `0xdf98`, the lack of an executed pre-compare head rewrite,
and the walling `a7=0x2450`. The final interpretation does not follow from them.
The overlooked third anchor at literal `0x2884` directly contradicts the
proposed fix:

```text
0x2986  a2 := 0x2278
0x2989  a2 := [a2]          ; current TCB
0x298b  [a2+0] := a3        ; current->frame := outgoing frame
0x2990  [0x2b60] := a3
0x2995  [0x2b64] := a3
```

At this point `a3=0x12048`, the outgoing init context. If `current` had already
been changed to `0x10dfc`, `0x298b` would overwrite the incoming task's frame
pointer:

```text
[0x10dfc+0] := 0x12048
```

That destroys its real frame pointer `0x15f18`. Both scheduler anchors would
then contain init's `0x12048`, and the later restore would select init, not the
incoming task.

The correct conclusion is narrower:

- The incoming frame `0x15f18` is a viable execution destination.
- The natural transition path walls while processing the outgoing init frame.
- The proposed current-only upstream pre-commit cannot establish the B1 state.
- The unresolved boundary is the old-to-new transition/callback/bootstrap
  contract, not proven current-update ordering.

## 1. Decisive third-anchor evidence

### 1.1 Literal pool

The relevant `+0x100` overlay literal pool is:

```text
VMA 0x2880 -> 0x3180
VMA 0x2884 -> 0x2278  (&current)
VMA 0x2888 -> 0x2b60
VMA 0x288c -> 0x2b64
```

Raw image verification:

```bash
xxd -g4 -e -s 0x2984 -l 16 \
  ../xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin
```

Expected output begins:

```text
00002984: 00002278 00002b60 00002b64 00003170
```

### 1.2 Runtime ownership invariant

The context-save code uses the still-current task to own the frame just built
from the live outgoing CPU state. The store sequence is:

```text
n=47517  pc=0x298b  [0x10f10] <- 0x12048
n=47519  pc=0x2990  [0x2b60]  <- 0x12048
n=47521  pc=0x2995  [0x2b64]  <- 0x12048
n=47985  pc=0x285d  [0x2278]  <- 0x10dfc
n=49435  pc=0x2a31  [0x2b64]  <- 0x15f18
n=49469  pc=0x2a82  [0x2b60]  <- 0x15f18
```

Reproduction:

```bash
XDNA_FW_PROBE=1 \
XDNA_FW_WATCH_ADDR=0x10f10,0x10dfc,0x2278,0x2b60,0x2b64 \
XDNA_FW_MAX=50000 \
cargo test --lib m2c_probe_addr_store_watch -- --nocapture
```

This ordering is not merely incidental. Before the outgoing context is attached,
`current` must identify the task whose live state was saved. Changing only
`current` earlier breaks that ownership invariant.

### 1.3 No timing of a bare pre-commit produces the claimed state

- **Before `0x298b`:** incoming `[0x10dfc+0]` is overwritten with outgoing
  `0x12048`.
- **Between `0x298b` and `0x2990`:** `a3` remains outgoing `0x12048`, so the
  anchors still seed outgoing.
- **After `0x2990`:** this is the observed old/new mismatch; it is no longer the
  proposed pre-seed mechanism.

Pre-committing only `current` cannot make `0x2990` store `0x15f18`. Doing that
also requires different control flow, a different `a3`, or a separate active-slot
update.

## 2. Better interpretation of `0x2b60` and `0x2b64`

The labels "head" and "tail" encouraged the inference that inequality is an
invalid queue state. The surrounding control flow instead looks like an
active/old versus candidate/new context transition:

```text
0x2a31  candidate := current->frame
0x2a36  a3 := active/old
0x2a38..0x2a73  restore active/old registers
0x2a7f  compare active/old, candidate/new
0x2a82  active := candidate
0x2a84  restore old stack pointer
0x2a86  invoke transition callback through restored old a7
0x2a89  a3 := active/new
...     process new task context/MMU state
0x2adc  jump back to 0x2a5d for another restore pass
```

Confidence in this interpretation is high, though the stripped firmware prevents
authoritative naming of the fields.

The important structural consequence is that `old != new` is intentionally
handled. It is not inherently an impossible state. The code commits the new
active frame, invokes something using the restored old context, reloads the new
frame, and later restores again.

## 3. Question 5A: hidden local head re-derive

### Verdict

The concrete hypothesis "an untaken branch directly rewrites `0x2b60` to the
incoming frame before the restore" is not supported.

An image-wide literal reference scan found exactly five references to the
`0x2888 -> 0x2b60` literal:

| PC | Use |
|---|---|
| `0x298d` | Seed, followed by the store at `0x2990` |
| `0x29ce` | Read on the interrupt arm |
| `0x2a33` | Restore read |
| `0x2a75` | Comparison read; unequal fallthrough stores at `0x2a82` |
| `0x2a89` | Post-callback reload |

The only stores are:

- `0x2990`: initial `active := outgoing`
- `0x2a82`: `active := candidate`, after the unequal comparison

The syscall arm has no CFG edge back to `0x2990`. Its bit-loop back-edge is only
`0x2a19 -> 0x2a0c`; exit is `0x2a05 -> 0x2a1c`. The picker and
`schedule_next` do not receive or derive the `0x2b60` address.

Reproduction:

```bash
XDNA_FW_PROBE=1 \
XDNA_FW_LIT_LO=0x2b60 XDNA_FW_LIT_HI=0x2b65 \
cargo test --lib m2c_probe_literal_xref -- --nocapture
```

This rules out the specific missed-direct-store theory. It does **not** prove the
iter37b pre-commit mechanism. A different entry contract, bootstrap path, saved
callback state, or loader mapping could still be the divergence.

## 4. What B1 proves and does not prove

B1 forces:

```text
[0x2b60] := current->frame = 0x15f18
```

immediately before the restore. It therefore:

1. Selects the incoming frame.
2. Makes the active/candidate comparison equal.
3. Skips the old-context transition callback.
4. Reaches the incoming task's saved entry.

The long clean run proves that `0x15f18` is runnable and that bypassing the old
transition advances execution. It does not prove that the firmware should
naturally move the `0x2a82` commit ahead of the callback, nor that hardware
normally forbids the unequal path.

B1 is a useful bypass experiment, not a validated fix.

## 5. What B2 proves and does not prove

B2 changes `current` back to init after the picker has already:

- selected `0x10dfc`,
- updated prior/current metadata,
- and performed related queue/picker operations.

It does not undo those effects, so it is not a symmetric alternative to B1.

The `0xe035` destination is nevertheless meaningful rather than random:

```text
0x08b043e4  retw.n
0x00003dfc  movi.n a2,0
0x00003dfe  retw.n
0x2000e035  break 1,15   ; main-returned/noreturn backstop
```

Thus B2 supports only this conclusion: restoring init is the wrong useful
destination, whereas entering `0x10dfc` advances the system. It gives no evidence
for the mechanism by which the switch should happen.

## 6. Question 5B: `0x2a86 -> 0xdf98`

### Confirmed in the current reconstructed image

`0x2a86` lies inside the manually registered `CTXSW_CALLEE` `+0x100` overlay,
so it fetches raw file offset `0x2b86`. Those bytes are:

```text
05 51 0b
```

The emulator decoder computes:

```text
word   = 0x0b5105
imm18  = 0x2d44
base   = ((0x2a86 + 4) & ~3) = 0x2a88
target = 0x2a88 + (0x2d44 << 2) = 0xdf98
```

Independent GNU Xtensa decoding agrees:

```bash
xtensa-lx106-elf-objdump -D -b binary -m xtensa \
  --start-address=0x2b86 --stop-address=0x2b89 \
  ../xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin
```

The raw-file PC/target are each `0x100` higher; subtracting the section delta
gives VMA `0x2a86 -> 0xdf98`.

Runtime also confirms:

```text
n=49471  pc=0x2a86  Call0 0xdf98
n=49472  pc=0xdf98  Callx8 a7
n=49473  pc=0x2450  Unknown word 0
```

Reproduction:

```bash
XDNA_FW_PROBE=1 XDNA_FW_MAX=49475 \
cargo test --lib m2c_probe_trace_to_wall -- --nocapture
```

### Residual loader uncertainty

The numeric target and emulator execution are solid. The stronger assertion
that real PSP placement necessarily exposes the same bytes at numeric VMA
`0xdf98` is not fully proven, because the PSP segment table is unavailable and
the repository's low-VMA piecewise overlays are empirically reconstructed.

The current base mapping selects:

```text
VMA 0xdf98 + 0x5c = raw 0xdff4 -> e0 07 00 -> Callx8 a7
```

A hypothetical local `+0x50` mapping would instead select raw `0xdfe8`, which
contains the guarded helper body currently associated with VMA `0xdf8c`.
There is no positive evidence that such a section exists; it would introduce a
new boundary inside an otherwise coherent helper family. It is a speculative
but concrete residual hypothesis that the current `+0x100`-only overlay probe
cannot test.

Assessment:

- Numeric caller target `0xdf98`: **very high confidence**.
- Emulator executes unguarded bytes there: **confirmed**.
- Real PSP placement necessarily maps those same unguarded bytes there:
  **strongly supported, not hardware-proven**.

## 7. Question 5C: `a7` and guarded-entry behavior

The runtime sequence is:

```text
0x2a63  a7 := [old_frame+0x1c]
...
0x2a7f  compare old,new
0x2a82  active := candidate       ; no a7 write
0x2a84  a1 := [old_frame+4]       ; no a7 write
0x2a86  Call0 0xdf98              ; call0 does not rotate the AR window
0xdf98  Callx8 a7
```

There is no opportunity for `a7` to change between restore and indirect call.
For init, it remains `0x2450`.

The helper body is:

```text
0xdf8c  move arguments
0xdf8e  a7 := [a7+12]
0xdf94  if a7 == 0, skip
0xdf98  Callx8 a7
```

The actual scheduler call enters at `0xdf98`; the guarded dereference and null
test are not executed. For init, `[0x2450+12]` is zero in the emulator, so the
guarded entry would skip.

Observed frame values include:

- init `0x12048`: `a7=0x2450`, saved resume PC `0x08b043e4`
- fresh task `0x15f18`: `a7=0`, entry PC `0x08b041bc`
- later frame `0x15e78` for the same task: `a7=0`
- other fresh frames `0x122f8` and `0x124f8`: `a7=0`

These observations prove that none of the sampled frames currently contains a
valid direct callback. They do **not** prove that no task can ever save one.
Fresh zeroed frames and a run containing no second current-changing switch are
not an exhaustive sample of the state immediately before a genuine cross-task
yield.

The statement "the hook walls for every task, therefore `old != new` can never
happen" is therefore an unsupported universal generalization.

## 8. Better-fitting alternative hypothesis

The instruction structure supports this model more directly than iter37b:

1. The syscall saves init's outgoing context.
2. While `current=init`, `0x298b` attaches the new frame to init.
3. Both transition slots are seeded to the outgoing frame.
4. The scheduler legitimately picks higher-priority `0x10dfc`.
5. The candidate slot is refreshed from `current->frame=0x15f18`.
6. The old frame is restored far enough to recover an old-context transition
   target and its arguments.
7. The unequal path commits the new active slot and invokes the transition
   callback associated with the outgoing context.
8. On callback return, the code reloads the incoming frame and completes the
   second restore pass.

The wall occurs at step 7 because init's recovered `a7` is a data pointer.

The remaining root-cause candidates are therefore:

1. **Bootstrap/adoption callback construction.** Init is an already-running boot
   context adopted by the scheduler; before its first yield, it should acquire a
   valid transition thunk but does not in the current execution.
2. **Bootstrap special path.** The init-to-first-task handoff should bypass or
   guard the callback through a different request/task-state/entry path.
3. **Register-window or exception-context divergence.** The value saved as the
   transition target differs because the emulator exposes the wrong architectural
   register window or exception state at the save boundary.
4. **Loader placement at the callback helper.** Real numeric `0xdf98` exposes a
   guarded implementation due to a still-undiscovered local VMA/LMA mapping.
5. **Earlier scheduler/task-state divergence.** Some earlier field causes the
   correct bootstrap or callback-initialization path not to run.

Current-only pre-commit is not on this list because it contradicts the frame
ownership store at `0x298b`.

## 9. Highest-value discriminator

Run the natural unequal transition while changing only the outgoing callback
target.

In a probe, replace init frame `[0x12048+0x1c]` with the address of a verified,
ABI-correct no-op windowed function (`entry; retw.n`). Do not modify:

- `[0x2278]`
- `[0x2b60]`
- `[0x2b64]`
- the branch at `0x2a7f`
- the call at `0x2a86`

Observe whether execution:

1. calls and returns through the thunk,
2. reloads `0x15f18` at `0x2a89`,
3. reaches the second restore pass at `0x2a5d`,
4. takes the equal/no-callback path on that pass,
5. and `rfe`s to `0x08b041bc`.

If it does, the natural unequal path is a valid switch protocol and the root
problem is init's transition-callback/bootstrap construction. This experiment is
strictly more discriminating than B1 because it preserves the firmware's
existing transition protocol instead of bypassing it.

If it does not, capture the exact post-callback path before changing another
variable.

## 10. Recommended investigation order

1. **Run the callback-thunk discriminator.** One state change, one question.
2. **Trace init adoption.** Work backward from the required outgoing callback
   contract through task-init/adoption and the final boot function, not from the
   wall forward again.
3. **Cross-image differential.** Compare the corresponding transition tail and
   initial-frame construction in `1502_00`, `17f0_10/11`, `17f1_10`, and
   `17f2_10`. Search by instruction pattern rather than absolute address.
4. **Pin the exact PSP placement.** Derive the low-VMA segment map from the PSP
   loader or another authoritative artifact if available. Specifically kill or
   confirm the `0xdf98 + 0x50` guarded-byte alias.
5. **Validate Xtensa system/window semantics against an independent oracle.** A
   minimal QEMU or official Zephyr Xtensa microtrace should verify `Syscall`,
   window selection, and the registers visible to the context save. Stock Zephyr
   is an architectural baseline, not proof of AMD MERT semantics.
6. **Only then implement a fix.** The implementation should reproduce a derived
   firmware/architecture contract, not synthesize a scheduler RAM write with no
   backing instruction or hardware mechanism.

## 11. Process assessment

The reverse engineering has produced substantial, high-quality instruction-level
evidence. The lack of convergence is mainly in the promotion of observations into
semantic conclusions:

- "This poke runs" has repeatedly become "hardware must create this state by
  this mechanism."
- A few frame samples have become universal scheduler invariants.
- Absence of behavior in the emulator has been used to infer intended hardware
  scheduling policy.
- Field names such as `head`, `tail`, and `hook` have acquired more authority
  than the surrounding code warrants.
- The append-only findings narrative retains superseded interpretations, making
  it easy for later reasoning to inherit an invalid premise.
- Empirical loader overlays are sometimes treated as hardware-proven placement
  even though the code itself records that the complete section layout remains
  unreconstructed.

A compact evidence ledger would help:

| Item | Classification | Rule |
|---|---|---|
| Exact bytes, decoded instruction, runtime value/store | Observation | May be used directly |
| Field/function name inferred from behavior | Interpretation | Must list competing readings |
| Claim about hardware path | Hypothesis | Must state a differentiating prediction |
| Poke result | Counterfactual experiment | Proves reachability/necessity, not mechanism |
| Proposed fix | Implementation | Requires a confirmed root-cause contract |

The immediate reset should be semantic, not mechanical: preserve the strong
trace evidence, discard the iter37b pre-commit conclusion, and make the callback
contract the next single-variable question.

## 12. Commands used for the adversarial audit

```bash
# Full transition function, runtime-selected overlay bytes
XDNA_FW_PROBE=1 XDNA_FW_DISASM=0x2730:0x2af0 \
cargo test --lib m2c_probe_disasm_range -- --nocapture

# Natural yield call graph
XDNA_FW_PROBE=1 \
XDNA_FW_CG_WARMUP=47350 XDNA_FW_CG_MAX=49500 XDNA_FW_CG_LINES=500 \
cargo test --lib m2c_probe_yield_callgraph -- --nocapture

# Syscall-to-context-save entry path
XDNA_FW_PROBE=1 XDNA_FW_MAX=47600 XDNA_FW_STOP_PC=0x290a \
cargo test --lib m2c_probe_trace_to_wall -- --nocapture

# Context-frame ownership and anchor ordering
XDNA_FW_PROBE=1 \
XDNA_FW_WATCH_ADDR=0x10f10,0x10dfc,0x2278,0x2b60,0x2b64 \
XDNA_FW_MAX=50000 \
cargo test --lib m2c_probe_addr_store_watch -- --nocapture

# Exhaustive direct literal references to the two anchors
XDNA_FW_PROBE=1 XDNA_FW_LIT_LO=0x2b60 XDNA_FW_LIT_HI=0x2b65 \
cargo test --lib m2c_probe_literal_xref -- --nocapture

# Caller and callback helper
XDNA_FW_PROBE=1 XDNA_FW_DISASM=0x2a75:0x2a8b \
cargo test --lib m2c_probe_disasm_range -- --nocapture

XDNA_FW_PROBE=1 XDNA_FW_DISASM=0xdf8c:0xdfa8 \
cargo test --lib m2c_probe_disasm_range -- --nocapture

# Yield wrapper itself
XDNA_FW_PROBE=1 XDNA_FW_DISASM=0x8b043cc:0x8b043e8 \
cargo test --lib m2c_probe_disasm_range -- --nocapture

# Independent decoding of the raw caller bytes
xtensa-lx106-elf-objdump -D -b binary -m xtensa \
  --start-address=0x2b86 --stop-address=0x2b89 \
  ../xdna-driver/amdxdna_bins/firmware/1502_00/npu.dev.sbin
```

## Final recommendation

Treat iter37b as **refuted**, not merely unconfirmed. Preserve these confirmed
facts:

- `0x2a86` numerically targets `0xdf98` in the reconstructed image.
- The executed helper directly performs `Callx8 a7`.
- Init supplies `a7=0x2450`, with no intervening rewrite.
- No hidden direct pre-compare store to `0x2b60` was found.
- Incoming frame `0x15f18` runs when selected.

Discard this conclusion:

> `current` must be changed to the incoming task before the context-save/head-seed
> path.

It conflicts with the firmware's explicit `current->frame := outgoing_frame`
store. The next investigation should preserve the unequal switch path and test
the outgoing callback contract directly.
