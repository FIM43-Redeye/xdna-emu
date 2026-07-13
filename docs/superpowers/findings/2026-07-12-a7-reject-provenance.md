# Phoenix `a7=6` reject provenance

Date: 2026-07-12  
Branch: `feat/m2c-mapping-boot-to-idle`  
Base commit: `67cbacf9`  
Image: Phoenix `1502_00/npu.dev.sbin`  
SHA-256: `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`

## Verdict

**VERIFIED: fork (A), genuine firmware state.** The rejected `a7=6` is the
current scheduler task's slot ID. The task at `0x10dfc` is initialized with ID
6, linked into runnable-table slot 6, selected as current by mapped firmware,
and later read through ordinary local data:

```text
SCHED                         = 0x2250
runnable[6]                   = [0x2250 + 0x38 + 6*4]
                              = [0x22a0] = 0x10dfc
SCHED.current                 = [0x2250 + 0x28]
                              = [0x2278] = 0x10dfc
SCHED.current->slot_id        = [0x10dfc + 0x08]
                              = [0x10e04] = 6
descriptor[5]                 = [0xfae0 + 0x14] = 6
sequencer argument a7         = 6
required predicate at 0x7fc7 = a7 < 6
```

The exact missing state is therefore **an in-range service context: the
current scheduler task presented to `0x26d4 -> 0xc530 -> 0x7fc4` must have a
slot ID in `0..=5`**. The reconstructed run instead enters that service path
while slot-6 task `0x10dfc` is current. No value is supplied by device SRAM or
by either `HARNESS_VIEW` transition.

`slot_id` is a structural name derived here, not an imported symbol: the same
field is used as the index into `SCHED+0x38`, and the selected table entry is
then installed at `SCHED+0x28`. Older names such as “column” or “priority” are
not needed for this verdict.

## Probe and fidelity boundary

The env-gated `m2c_probe_26d4_cache_pageroot_timeline` observer now records:

- AR-producing loads, their effective addresses, values, and memory classes;
- stores in the bounded producer/consumer ranges;
- register-window rotations through `0x26d4`, `0xc530`, and `0x7fc4`;
- exact transition timelines for local words `0x10e04` and `0x2278`.

The instrumentation reads processor and local-memory state only. It does not
change firmware memory, registers, interrupts, MMIO, framing, or production
behavior. `HARNESS_VIEW` events remain explicitly labeled as not firmware.

**VERIFIED:** the image bytes at the principal decoded instructions match the
executed trace. Under the Phoenix BASE file offset `+0x5c`, examples include
`0xd4ef -> file 0xd54b: 39 28` (`S32iN`), `0x2728 -> file 0x2784: f8 28`
(`L32iN`), `0xc56c -> file 0xc5c8: fd 07` (`MovN`), and `0x7fc7 -> file
0x8023: f6 67 21` (`Bgeui`). No `17f1_10` byte or semantic was used.

## Backward provenance from the reject

All entries below are **VERIFIED** by the executed Phoenix trace and guarded by
the env-gated test unless a narrower confidence marker is shown.

| n | PC | Executed instruction / transition | Backward edge |
|---:|---:|---|---|
| 53873 | `0x7fc7` | `Bgeui a7,6,0x7fec`, `a7=6` | Rejects because the unsigned range predicate `a7 < 6` is false. |
| 53872 | `0x7fc4` | `Entry a1,0x20`, window base `5 -> 7` | The caller's outgoing `a15=6` becomes sequencer `a7=6`. |
| 53870 | `0xc56c` | `MovN a15,a7`, `a7=6` | `0xc530` forwards its live-in argument unchanged. |
| 53825 | `0xc54d` | `S32iN a7,[a10+0x14]` | Also records the same value as descriptor word 5: `[0xfaf4]=6`. |
| 53814 | `0xc530` | `Entry a1,0x30`, window base `3 -> 5` | The `0x26d4` caller's outgoing `a15=6` becomes wrapper `a7=6`. |
| 53807 | `0x2728` | `L32iN a15,[a8+8]` | Loads `[0x10e04]=6` from ordinary local data. |
| 53804 | `0x2720` | `L32iN a8,[a7+40]` | Loads `[0x2278]=0x10dfc`, the current-task pointer, from ordinary local data. |
| 53801 | `0x2718` | `L32r a7,[0x25f0]` | Fetch literal gives `SCHED=0x2250`; this is an instruction literal, not the rejected data. |
| 47985 | `0x285d` | `S32iN a2,[a7+40]` | Mapped firmware writes `[0x2278]=0x10dfc`; this occurs before either `HARNESS_VIEW`. |
| 47984 | `0x285b` | `L32iN a2,[a2+56]` | Loads the selected pointer from `[0x22a0]=0x10dfc`. |
| 47968-47969 | `0x282b..0x282e` | `Addx4 a4,a3,a4`; `L32iN a5,[a4+56]` | With selector `a3=6` and `SCHED=0x2250`, computes slot address `0x22a0` and reads task `0x10dfc`. |
| 39852 | `0xd60f` | `S32iN a8,[a3+56]` | Task initialization linked `0x10dfc` at `[0x22a0]`. |
| 39760 | `0xd538` | `Addx4 a3,a3,a15` | Converts initializer argument 6 into `SCHED + 6*4 = 0x2268`; the later `+56` is slot `SCHED+0x38+6*4`. |
| 39730 | `0xd4ef` | `S32iN a3,[a8+8]` | Mapped firmware creates the source word: `[0x10e04] 0 -> 6`. |

The exact transition watches close both memory edges:

```text
0x10e04:
  n=0      0
  n=39730  pc=0xd4ef  0 -> 6
  no further change through the reject

0x2278:
  n=0      0
  n=41463  pc=0x46ac  0 -> 0x10f10
  n=47985  pc=0x285d  0x10f10 -> 0x10dfc
  no further change through the reject
```

The first code-view transport is later, at `n=53640` (`HARNESS_VIEW_SPLIT_8cxx`);
the BASE `0x26d4` view is selected at `n=53784`. The task-ID producer and the
current-task selection therefore both predate those transports and execute as
mapped firmware. The earlier `n=47672` `HARNESS_OBSERVER` only disables an
emulator fill-loop fast path to expose individual retired stores; it is marked
not firmware and is architecturally equivalent. It does not supply either
watched word.

## Controls

### First call does not carry 6

**VERIFIED:** the same mapped wrapper/sequencer chain executes before the later
BASE `0x26d4` entry:

```text
n=53629  0xc56c  MovN a15,a7    a7=0, a15 becomes 0
n=53632  0x7fc7  Bgeui a7,6     a7=0, reject not taken
```

The later call differs at the traced task-field load, not at the bounds check.

### `0x8c6c` is not the producer

**VERIFIED:** inside the first service call, `0x8c80 MoviN a7,1` changes only
the callee window. At `n=53672`, `0x8cba RetwN` rotates back and caller `a7`
returns to 0 (the return value appears in the caller's `a15`). The final
`a7=6` is loaded later by the separate `0x26d4` path.

### Source classification

| Source | Value | Classification | HARNESS_VIEW-produced? |
|---|---:|---|---|
| `[0x22a0]`, runnable slot 6 | `0x10dfc` | ordinary local data | No; linked at `n=39852`. |
| `[0x2278]`, current task | `0x10dfc` | ordinary local data | No; selected at `n=47985`. |
| `[0x10e04]`, task `+0x08` | `6` | ordinary local data | No; initialized at `n=39730`. |
| `[0xfaf4]`, descriptor word 5 | `6` | ordinary local data | No; downstream copy at `n=53825`. |
| `[0x25f0]`, SCHED literal | `0x2250` | instruction literal | The later read is in the BASE instruction view, but it is not the rejected value. |
| device SRAM/MMIO | none | not in provenance | No. |

The consumer instruction at `0x2728` is reached through the counterfactual
`HARNESS_VIEW_BASE_26d4` transport. That is a known execution-transport
boundary, but it does not synthesize `a7`: the loaded data and both of its
firmware writers are independently observed before either view selection.
Under the brief's source-memory discriminator this is fork (A), not fork (B).

## What the firmware expects

**VERIFIED:** the immediate contract is `descriptor[5] < 6`, enforced by
`Bgeui a7,6` before the sequencer body. **VERIFIED:** this descriptor field is
fed by `SCHED.current->slot_id` in the observed path. Thus the precise state
that is absent is:

> `SCHED.current` names a runnable task whose slot ID is in the sequencer's
> accepted set `0..=5`.

The actual state is `SCHED.current=0x10dfc` and
`SCHED.current->slot_id=6`. This finding does not claim why the firmware chose
slot 6, and it does not assign an undocumented “column” or “priority” meaning
to the field. It establishes the structural scheduler identity and the exact
range contract.

This state is derived, not calibrated: trace the producer of the selector that
becomes `a3=6` in the mapped `0x2800..0x2878` scheduler body, together with the
runnable/current tables above. On hardware, the same contract could be checked
with an instruction/data trace or halted management-core snapshot of these
words. The existing host aperture cannot directly read management local RAM,
so there is no safe host one-shot MMIO read that closes it; a downstream
service-transaction trace is the non-invasive hardware alternative. No value
should be injected.

## Ranked single next step

**Trace the mapped caller that supplies selector 6 to the `0x2800..0x2878`
scheduler body.** The present trace already pins its consumption:

```text
n=47960  0x2816  L32iN a6,[a1+0x14] -> 6
n=47966  0x2826  MovN a10,a6        -> 6
n=47967  0x2828  Extui a3,a10,0,8   -> 6
n=47968  0x282b  Addx4 ...          -> SCHED + 6*4
```

Walking that stack argument to its mapped caller will distinguish an intended
slot-6 service attempt from an earlier scheduler/lifecycle divergence. The
closing evidence is a complete producer chain for that selector, compared with
the successful first-call selector 0. This remains derive-only: no branch,
task word, current pointer, or descriptor value is forced.

## Verification

Fresh results for this uncommitted diff:

```text
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
  > build/experiments/firmware-re/a7-reject-provenance.log 2>&1

cargo test --lib
```

The probe is test-only and env-gated. Production `load_m2c`, MMIO, and Xtensa
system behavior are unchanged. Nothing was committed by this pass.
