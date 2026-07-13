# Phoenix slot-6 selector provenance

Date: 2026-07-12  
Branch: `feat/m2c-mapping-boot-to-idle`  
Base commit: `d5824e21`  
Image: Phoenix `1502_00/npu.dev.sbin`  
SHA-256: `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`

## Verdict

**VERIFIED: the selector provenance supports fork (2), an intended slot-6
worker selection, not an upstream reconstruction divergence.** A pending
go-alive queue record sets the scheduler's eligibility gate. The mapped
Phoenix scheduler scans to runnable index 6, matches two record fields against
two fields of task `0x10dfc`, writes loop index 6 to its own frame, and makes
`runnable[6]` current. Every producer is mapped firmware, an instruction
constant, or ordinary local data written by mapped firmware. The chain contains
no `HARNESS_VIEW` byte, wrong decode, off-by-one count, or forced value.

The brief's description of `[a1+0x14]` as an argument "supplied by the caller"
is too strong. It is a scheduler-frame local in both observed invocations. The
scheduler itself writes 6 at `0x27bc`; the caller influences the result through
the queued record and queue count, not by passing selector 6 in a register or
stack argument.

The concrete discriminator is the fixed-pool count at `SCHED+0x274 = 0x24c4`:

```text
first scheduler invocation:   [0x24c4]=1 -> eligibility flag a5=1 -> select 6
queue pop:                     [0x24c4]=1 -> 0
second scheduler invocation:  [0x24c4]=0 -> eligibility flag a5=0 -> skip 6
```

**VERIFIED: this is not the old ~131-entry go-alive livelock.** The guarded
sequence is:

```text
scheduler selectors = [6, 0]
service a7 values    = [0, 6]
0x55f8 entries       = [n=49925]
```

Slot 6 is selected once. The next scheduler invocation reaches index 6 again
but skips it and retains default selector 0. The one go-alive dispatch executes
the alive publisher at `0x50c6..0x50cf` before either service-range outcome.
The prior ~131-entry recurrence was the already-closed mixed queue-tail framing
artifact that left an output descriptor valid; it is absent with the coherent
queue tail in this baseline.

Fork (2) therefore needs one qualification: the trace proves that selecting the
slot-6 worker is intended by the mapped scheduler. It does **not** promote the
later `0x7fec` abort to a hardware-ground-truth path. That later call is reached
only after the probe's explicit `HARNESS_VIEW` counterfactuals. The clean,
pre-counterfactual worker path has already published alive; the remaining known
gap is where VA `0..3` lands in the emulator, not whether slot 6 was selected.

## Probe and fidelity boundary

The env-gated `m2c_probe_26d4_cache_pageroot_timeline` observer now adds:

- dynamically derived selector-frame effective addresses and their writers;
- the queue-count producer/load/retirement timeline at `0x24c4`;
- bounded mapped traces for the queue enqueuer, scheduler scan, and queue pop;
- an exact transition watch for the second invocation's selector word
  `0x30c4`;
- the ordered selector, service-slot, go-alive-entry, and publisher-store
  sequences.

The additions only read registers, decoded store operands, local memory, and
MMU state. They do not modify firmware data, CPU state, MMIO, branches, or
production mappings. Existing `HARNESS_VIEW` transports remain explicitly
labeled as not firmware.

**VERIFIED:** the Phoenix image hash above was recomputed. Raw image bytes match
the executed decoder at the decisive mapped instructions, including
`0x26c8 -> file 0x27c8: 22 02 74`, `0x2720 -> file 0x2820: 1b 22`,
`0x278d -> file 0x288d: 07 e8 8f`, `0x27b7 -> file 0x28b7: 40 f2 83`,
`0x27bc -> file 0x28bc: f9 51`, `0x2816 -> file 0x2916: 68 51`,
`0xccae -> file 0xcdae: 22 4a 74`, and
`0xd785 -> file 0xd885: c4 8c 3a`. The Segment-B fill instruction at
`0x08b0e29a` matches file `0x3b39a: 32 45 00`. No `17f1_10` byte or semantic
was used.

## Selector-6 producer chain

All rows are **VERIFIED** by the executed Phoenix trace. “Ordinary local” means
management-core local data, not device SRAM, host memory, or a harness-supplied
code byte.

| n | PC | Producer or consumer | Value and source class | Backward edge |
|---:|---:|---|---|---|
| 47960 | `0x2816` | `L32iN a6,[a1+0x14]` | `[0x30e4]=6`, ordinary local | Consumes the selector written in this same scheduler frame. |
| 47886 | `0x27bc` | `S32iN a15,[a1+0x14]` | `[0x30e4] <- 6`, ordinary local | The direct selector producer; no caller writes this word. |
| 47884 | `0x27b7` | `Moveqz a15,a2,a4` | `a15: 0 -> 6` | `a4==0`, so the candidate becomes loop index `a2=6`. |
| 47882 | `0x27b1` | `Sub a4,a4,a9` | `1 - 1 -> 0` | The candidate's byte at task `+0x0c` equals the queued record value saved at frame `+0x10`. |
| 47875 | `0x279d` | `L32iN a9,[a1+0x10]` | `[0x30e0]=1`, ordinary local | Reads the queued record value saved by `0x2704`. |
| 47873 | `0x2797` | `L8ui a4,[task+0x0c]` | `[0x10e08]=1`, ordinary local | Reads task `0x10dfc`'s matching field. |
| 47870-47871 | `0x2790..0x2793` | task/frame byte loads | `[0x10e2b]=4`, `[0x30f4]=4`, ordinary local | A second queued-record/task criterion also matches. |
| 47869 | `0x278d` | `Bbsi a8,0,0x2720` | not taken | The queue-enabled gate permits candidate evaluation. |
| 47867 | `0x2787` | `Xor a8,a5,a8` | `0xffffffff -> 0xfffffffe` | `a5=1` clears bit 0 of the observed gate expression. |
| 47849 | `0x2720` | `AddiN a2,a2,1` | `5 -> 6` | The firmware's own scan produces index 6; both invocations execute this increment. |
| 47692 | `0x2704` | `S32iN a4,[a1+0x10]` | `[0x30e0] <- 1`, ordinary local | Copies queued record byte `[0x2330]`. |
| 47684 | `0x26f3` | `S32iN a15,[a1+0x24]` | `[0x30f4] <- 4`, ordinary local | Copies low four bits of queued record byte `[0x232b]`. |
| 47681-47683 | `0x26ea..0x26f0` | record loads and `Extui` | `[0x232b]=4`, `[0x2330]=1`, ordinary local | Loads the current fixed-pool record selected by cursor 0. |
| 47678 | `0x26e2` | `MoviN a5,1` | instruction constant | Executed only on the non-empty queue path. This is the decisive eligibility input at `0x2787`. |
| 47677 | `0x26df` | `L8ui a2,[0x24c5]` | cursor `0`, ordinary local | Selects record 0 at `0x2320..0x2334`. |
| 47668 | `0x26c8` | `L8ui a2,[0x24c4]` | count `1`, ordinary local | Makes `Beqz @0x26da` fall through to the record-load path. |
| 47383 | `0xd785` | `S8i a4,[0x24c4]` | count `0 -> 1`, ordinary local | Mapped queue enqueuer publishes the record. |
| 47368 | `0xd6f7` | `S8i a2,[0x232b]` | record byte `4`, ordinary local | Mapped queue enqueuer creates the first match input. |
| 47361 | `0xd6e3` | `S8i a14,[0x2330]` | record byte `1`, ordinary local | Mapped queue enqueuer creates the second match input. |
| 39748-39749 | `0xd519..0xd51c` | task-field stores | `[0x10e2b] <- 4`, `[0x10e08] <- 1`, ordinary local | Mapped task initializer creates the matching task fields. |
| 39852 | `0xd60f` | runnable-table store | `[0x22a0] <- 0x10dfc`, ordinary local | Links that task at `SCHED+0x38+6*4`. |

The direct register-level computation is therefore:

```text
queue count 1
  -> non-empty path
  -> a5 = 1
  -> slot-6 gate at 0x278d is not taken

record[+0x0b] = 4 == task[+0x2f] = 4
record[+0x10] = 1 == task[+0x0c] = 1
  -> subtraction result 0
  -> Moveqz chooses loop index 6
  -> [frame+0x14] = 6
```

After the scan, `0x2816..0x282e` reloads 6, indexes
`SCHED+0x38+6*4 = 0x22a0`, and reads task `0x10dfc`. At `n=47985`, mapped
firmware stores that pointer to `SCHED.current = [0x2278]`. Those downstream
edges remain as established in the predecessor finding.

There is no off-by-one discriminator here. The loop reaches index 6 in both
invocations. Eligibility, not the loop bound, decides whether index 6 becomes
the selector.

## Selector-0 contrast

The “successful first call” at `n=53629/53632` is the first **service** call,
but its selector comes from the second observed scheduler invocation. Keeping
those two orderings separate removes an apparent contradiction.

| Edge | Selector-6 invocation | Selector-0 invocation |
|---|---|---|
| scheduler entry | `n=47564`, return PC `0xdd30` | `n=53155`, called at `0xc66c` |
| queue count at `0x26c8` | `[0x24c4]=1` | `[0x24c4]=0` |
| path at `0x26da` | falls through; reads record | branches to `0x26f5`; skips record loads |
| gate input `a5` | `1` at `0x26e2` | `0` at `0x26d1` |
| loop at slot 6 | `a2: 5 -> 6` at `n=47849` | `a2: 5 -> 6` at `n=53459` |
| `Xor @0x2787` | `0xffffffff -> 0xfffffffe` | remains `0xffffffff` |
| `Bbsi @0x278d` | not taken; evaluates candidate fields | taken directly to `0x2720`; skips candidate fields |
| frame selector | `0x27bc` writes `[0x30e4]=6` | no selector write; `[0x30c4]` remains zero |
| `L32iN @0x2816` | loads `6` at `n=47960` | loads `0` at `n=53552` |
| runnable lookup | `[0x22a0]=0x10dfc` | `[0x2288]=0` |

The queue pop closes the causal edge between the two invocations:

```text
n=49686  0xcc25  load  [0x24c4] -> 1
n=49734  0xcca9  reload[0x24c4] -> 1
n=49735  0xccac  AddiN          -> 0
n=49736  0xccae  store [0x24c4] <- 0
n=53259  0x26c8  next scheduler load -> 0
```

All are mapped firmware over ordinary local data. This is correct queue
retirement, not the earlier mixed-tail stale-descriptor failure.

Selector 0 is a default/fallback value, not an explicit winning slot-0 task.
The second frame word has this exact provenance:

```text
n=0      [0x30c4] = 0
n=2928   pc=0x4525       0 -> 0xdeadbeef
n=7279   pc=0x08b0e29a  mapped firmware fill -> 0
n=53552  pc=0x2816      load -> 0
```

The fill loop is collapsed by the emulator's existing fast path, so the exact
word watch records its architectural transition rather than claiming a
separately retired byte store for `0x30c4`. Its instruction bytes come from the
Phoenix image, and the effect occurs thousands of instructions before any
`HARNESS_VIEW` transition. No later writer changes the word. With no eligible
candidate, `0x2816` reads that default zero, the scheduler indexes empty
`runnable[0]`, and the downstream wrapper forwards `a7=0` at `n=53629`; the
range check at `n=53632` falls through.

## Source classification and fork decision

| Input | Observed value | Class | Firmware producer | Harness-supplied? |
|---|---:|---|---|---|
| queue count `[0x24c4]` | `1`, later `0` | ordinary local | `0xd785`, later `0xccae` | No |
| queue cursor `[0x24c5]` | `0` | ordinary local | mapped queue setup | No |
| record `[0x232b]` | `4` | ordinary local | `0xd6f7` | No |
| record `[0x2330]` | `1` | ordinary local | `0xd6e3` | No |
| task `[0x10e2b]` | `4` | ordinary local | `0xd519` | No |
| task `[0x10e08]` | `1` | ordinary local | `0xd51c` | No |
| scan index | `6` | register arithmetic | `AddiN @0x2720` | No |
| selector `[0x30e4]` | `6` | ordinary local | `Moveqz @0x27b7`, store `@0x27bc` | No |
| fallback `[0x30c4]` | `0` | ordinary local | mapped early fill effect | No |
| device SRAM/MMIO | none | not in selector cone | none | No |

The first code-view counterfactual is later at `n=53640`
(`HARNESS_VIEW_SPLIT_8cxx`); the BASE `0x26d4` view is selected at `n=53784`.
The queue enqueue, selector-6 computation, current-task write, one go-alive
dispatch, alive publisher, queue retirement, and selector-0 computation all
precede those view transports. Under the brief's discriminator there is no
“earliest wrong input” to name: fork (1) is not supported.

## Intended slot-6 completion and alive publication

The slot-6 worker does not wait for the later range-check call to publish. Its
one queued run function executes this clean mapped path:

```text
n=49925  0x55f8  go-alive run-function entry
n=51765  0x560d  Call8 0x5044
n=51766  0x5044  publisher entry
n=52115  0x50ba  L32r destination base -> 0
n=52119  0x50c6  S8i [0] <- 0x00
n=52120  0x50c9  S8i [3] <- 0x03
n=52121  0x50cc  S8i [2] <- 0x0b
n=52122  0x50cf  S8i [1] <- 0xb0
```

The stores compose little-endian `0x030bb000` at firmware VA `0..3`. This is
the intended completion/publish path requested by fork (2), and it executes
before the first accepted service guard (`a7=0`, `n=53632`) and before the
later slot-6 abort (`n=53873`). The prior alive finding already bounded the
remaining discrepancy: the emulator leaves these writes at local PA `0..3`,
whereas host visibility requires the established `FW_ALIVE_OFF` destination.
This pass does not reopen the closed PSP-loader or below-CPU-bank mechanism
hunts.

The later `a7=6` call is still genuine scheduler data: after slot 6 becomes
current, the counterfactual BASE `0x26d4` path reloads
`[0x2278]=0x10dfc`, then `[0x10e04]=6`, and reaches
`Bgeui @0x7fc7 -> 0x7fec`. But temporally it cannot be the cause of either
missing publication or repeated go-alive dispatch: publication already ran,
and no second `0x55f8` entry occurs.

## Livelock result

**VERIFIED: the scheduler does not re-select slot 6 on each dispatch.** It
selects 6 once while the queue count is 1, the worker consumes the single
record, and the next scheduler pass produces fallback 0 after the count reaches
0. `SCHED.current` remains task `0x10dfc`, but persistence of the current-task
pointer is not re-selection.

Thus the brief's proposed link

```text
slot-6 reject -> reselect slot 6 -> redispatch 0x55f8 (~131x)
```

is false in the current trace. The observed sequence is instead:

```text
enqueue once -> select worker 6 -> dispatch 0x55f8 once -> publish alive
             -> retire queue -> fallback selector 0 -> service a7=0
             -> later counterfactual reload of current slot 6 -> 0x7fec
```

## Ranked single next step

**After review, update the arc's durable handoff to supersede the predecessor's
“missing in-range service context” conclusion and bank the selector branch as
closed.** Carry forward the verified frontier exactly: one intended slot-6
worker selection, one `0x55f8` dispatch, and the four publisher stores at
`0x50c6..0x50cf`. Do not reopen selector logic, PSP-loader RE, the `0x8cae`
mechanism, or below-CPU banking without a genuinely new derivable source. This
is the shortest derive-only next move and prevents the stale reject framing
from routing another investigation back into a closed branch.

## Verification

Fresh results for this uncommitted diff:

```text
XDNA_FW_PROBE=1 cargo test --lib \
  m2c_probe_26d4_cache_pageroot_timeline -- --nocapture

targeted provenance probe:
  1 passed; 0 failed; 4120 filtered out

cargo test --lib:
  4091 passed; 0 failed; 30 ignored

cargo fmt --all -- --check:
  exit 0

git diff --check:
  exit 0
```

## Reproduction

```text
XDNA_FW_PROBE=1 cargo test --lib \
  m2c_probe_26d4_cache_pageroot_timeline -- --nocapture \
  > build/experiments/firmware-re/slot6-selector-provenance.log 2>&1

cargo test --lib
```

The probe is test-only and env-gated. Production `load_m2c`, MMIO, and Xtensa
system behavior are unchanged. Nothing was committed by this pass.
