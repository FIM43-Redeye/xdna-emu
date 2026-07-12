# The dissolving split is globally wrong; an EXCM loop bug amplified its budget tail

Date: 2026-07-12  
Target: Phoenix/NPU1 firmware `1502_00/npu.dev.sbin`  
SHA-256: `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`

## Verdict

Primary cause **#2: wrong upstream data from a globally wrong framing**. A real
cause-#1 interpreter bug amplified the symptom but did not cause the non-publish.

The candidate

```text
delta_lo=+0x100 delta_hi=0 split=0x8cae literal_delta=+0x5c
```

dissolves the shared-byte contradiction only by splicing unrelated instruction
streams. The splice eventually writes `PS=0x27261114`, including `PS.EXCM=1`.
Xtensa suppresses zero-overhead-loop back-edges in exception mode, but the
interpreter omitted that gate in its ordinary, FLIX, and fill-fastpath loop
tails. That bug made each `0xc961` activation execute 32 passes instead of one.

After the faithful EXCM fix, the candidate still builds `_NPU`, enters the
service, never writes either device-SRAM landmark, and consumes a full
1,000,000-instruction budget. Each `0xc961` activation now executes one body
pass, reaches `0xc972`, and returns into a recurring 145-instruction scheduler
cycle.

The bad framing is already causal before the service call. It diverts the
AT-framed publisher at `0x8cac`, later causes `0x94ea` to overwrite the local
scheduler input at `0x2c`, and makes `sched_ready_popcount` return zero. The
same candidate then executes a separately incoherent service epilogue that
writes `PS=0x27261114` instead of restoring the saved `0x00060022`.

Restoring the coherent BASE service bytes revives the original publisher
collision. There is therefore no `load_m2c` correction to present from this
candidate.

## 1. The LOOP bug is real but secondary

The live function is BASE-framed `sched_ready_popcount` at `0xc938`:

```text
0xc95b  Movi  a5,32
0xc95e  Movi  a4,0
0xc961  Loop  a5,0xc972
0xc964  Extui a5,a3,0,1
0xc967  AddN  a4,a5,a4
0xc969  Extui a5,a4,0,8
0xc96c  Bgeui a5,2,0xc979
0xc96f  Srli  a3,a3,1
0xc972  Wsr   PS,a2
0xc975  MoviN a2,0
0xc977  RetwN
0xc979  Wsr   PS,a2
0xc97c  MoviN a2,1
0xc97e  RetwN
```

`Loop a5` correctly writes `LCOUNT=a5-1`, so `a5=32` arms `LCOUNT=31`. The
architectural loop-end rule then checks `PS.EXCM`: if it is set, execution must
fall through at `LEND` without taking or consuming a back-edge.

The independent Xtensa SLEIGH semantics in
`ghidra/Ghidra/Processors/Xtensa/data/languages/xtensaMain.sinc:284-289`
state the same rule directly: at a loop end, `LCOUNT == 0 || PS.EXCM` falls
through; only the remaining case decrements `LCOUNT` and jumps to `LBEG`.

Before the fix, a 100,000-instruction candidate trace reported:

```text
calls=130 loops=130 iterations=4160 exhausted=130 early=0
```

Those 32-pass activations and the old 341-instruction spacing are evidence of
the missing EXCM gate, not faithful firmware behavior. The candidate's
delta-zero service tail had already set `PS=0x27261114`; bit 4 is EXCM.

Three test-first regressions failed on the old implementation:

- ordinary not-taken fallthrough at `LEND` redirected to `LBEG` under EXCM;
- the independent FLIX loop tail did the same;
- the fill-loop fastpath collapsed all remaining iterations under EXCM.

The fix adds `!PS.EXCM` to both loop-end tails and makes the fastpath decline in
exception mode. Post-fix candidate execution now shows exactly one
`0xc964..0xc96f` body pass followed by `0xc972`; `LCOUNT` remains 31 because no
back-edge is taken. The service-tail hits after the first pass occur at
`n=55709,55854,55999,56144`, exactly 145 instructions apart. Thus the LOOP bug
was a real multiplier, while repeated execution of `0xc961` still comes from
ordinary outer re-entry.

## 2. The outer recurrence and its exact input

`0xd836` calls `0xc938`. The helper builds its bitmap as follows:

```text
0xc93e  L32r a4,[0x3d28]       -> 0x2250
0xc941  MoviN a5,6
0xc943  Movi a3,0
0xc946  Addi a4,a4,0x38        -> 0x2288
0xc949  Loop a5,0xc95b         -> six slots, 0x2288..0x229c
0xc94c  L32iN a5,[a4]          -> task pointer
0xc94e  L8ui a6,[a5+0x2c]      -> state
0xc951  L32iN a5,[a5+0x38]     -> mask
0xc953  Bnei a6,1,0xc959
0xc956  Or a3,a5,a3
```

It returns 1 only when the OR of masks from state-1 entries has at least two
set bits. In the dissolving candidate all six slot words are zero. With EXCM
faithfully suppressing the six-slot loop's back-edge, the helper examines the
first zero slot and its field reads are from local `0x2c` and `0x38`:

```text
[0x2288..0x229c] = {0,0,0,0,0,0}
[0x2c]            = 0xb7 (183, not state 1)
[0x38]            = 0xb0cd13b7
a3 at 0xc961      = 0
result at 0xd839  = 0
idle target 0xd842 = 0x588c
```

`0xd842` calls the idle/run function at `0x588c`; that path reaches
`0x7fe1 -> 0x8c6c` again, returns to `0xd7f0`, and repeats. This is the full
post-fix 145-instruction recurrence. It is not one stuck activation inside
`0xc961`.

## 3. Where the candidate corrupts that input

The split is inside a three-byte publisher instruction. The coherent AT bytes
and the candidate's mixed bytes are:

```text
coherent AT:  0x8cac [87 ba 02]  Bgeu ...,target=0x8cb2
candidate:    0x8cac [87 ba d2]  Bgeu ...,target=0x8c82
```

The first two candidate bytes come from `VMA+0x100`; byte `0x8cae` comes from
`VMA+0`. The resulting backward branch enters the BASE-framed service prefix
and returns through the delta-zero tail instead of executing the publisher's
coherent AT body through `0x8d50`.

That wrong path later executes the store loop at `0x94ea`. The first pass fills
local words through `0x3c` with `0xb0cd13b3`; firmware subsequently restores
`[0x2c]=1` at `0xc8c7`. A second pass then overwrites the same band with
`0xb0cd13b7`:

```text
n=54225 pc=0x94ea  [0x2c] <- 0xb0cd13b7
n=54255 pc=0x94ea  [0x38] <- 0xb0cd13b7
```

The exact second timestamp follows the loop's observed ten-instruction word
stride; the retained trace prints through `[0x34]` before its 64-event cap and
the service-edge read confirms `[0x38]=0xb0cd13b7`.

The coherent AT publisher does not execute this fill. At the otherwise
identical service edge it has:

```text
[0x2c] = 1
[0x38] = 0x90cd1530
```

Under the existing BASE-service counterfactual, `0xc938` ORs
`0x90cd1530`, takes the `>=2` exit, returns 1 at `0xc97c`, and `0xd839` branches
past the idle call. Execution reaches `0x7fe7` and then the separately known
`0x26d4` view seam. This control falsifies cause #3 for the candidate: coherent
upstream data makes the scheduler test pass immediately.

## 4. The candidate service tail is independently incoherent

At the candidate's natural `0x8c6c` service entry, the BASE-framed prefix
produces these live values:

```text
0x8c6f Rsil a3,2       a3 = saved PS 0x00060022
a4 = 0x27271114
a5 = 0x000117b0
a6 = 0x000000f7
```

The correct BASE tail is:

```text
0x8cae Addi  a8,a8,0x60
0x8cb1 Addmi a4,a4,0x1000
0x8cb4 Addmi a5,a5,0x1000
0x8cb7 Wsr   PS,a3
0x8cba RetwN
```

The candidate's delta-zero tail instead decodes:

```text
0x8cae L8ui  a13,[a6]
0x8cb1 MoviN a14,8
0x8cb3 MoviN a15,0
0x8cb5 Or    a3,a13,a14
0x8cb8 S8i  a3,[a6]
0x8cbb S32i a15,[a5]
0x8cbd Wsr   PS,a4
0x8cc0 RetwN
```

Its service-side literal framing is wrong as well. `L32r a5` at `0x8c78`
targets VMA `0x3560`; the candidate leaves `[0x3550,0x3564)` at AT, so it reads
file `0x3660` and obtains `0x000117b0`. The coherent BASE service view reads
file `0x35bc` and obtains the device pointer `0x27271000`.

It therefore treats `0xf7` as an address, clears `[0x117b0]`, overwrites the
saved `a3`, and writes `PS=0x27261114`. Bit 4 (`0x10`) is set, so this is also
the exact instruction/data source of the secondary EXCM-loop symptom. The first
dispatcher entry is observed with exactly that invalid PS value. This is a
second, independent proof that delta zero is not the service framing.

## 5. Consequence for the collision search

Changing `delta_hi` from zero to the correct BASE `+0x5c` does not repair this
static split. It changes the same cross-split publisher instruction to:

```text
0x8cac [87 ba 82]  Bgeu ...,target=0x8c32
```

The publisher then stops at `Unknown 0x8c32`, never builds `_NPU`, and never
enters the service. Keeping AT supplies the coherent publisher but recreates
the known BASE-service collision.

So the locally dissolving assignment is rejected, but this does not produce a
new globally coherent assignment or a `load_m2c` diff. The hardware-derived
device-SRAM oracle remains red: no descriptor store reaches `0x030bb000`, and
no alive pointer store reaches `0x030bf000`. With the EXCM correction in place,
a fresh 1,000,000-instruction run still reports `publisher_pass=true`,
`service_entered=true`, `service_pass=false`, and `stop_kind=budget` at
`0xd83e`. The helper is no longer over-iterating; the budget is consumed by
re-entering the same failed scheduler test every 145 instructions.

## 6. Verification

The three EXCM regressions pass after the interpreter change:

```text
cargo test --lib loop_back_is_suppressed_in_exception_mode
  2 passed; 0 failed

cargo test --lib fill_loop_is_not_fast_pathed_in_exception_mode
  1 passed; 0 failed
```

The complete library suite remains green:

```text
cargo test --lib
  4088 passed; 0 failed; 30 ignored
```

The production-map SRAM probe remains an expected red oracle, rather than a
claimed publish fix:

```text
XDNA_FW_PROBE=1 XDNA_FW_MAX=100000 \
  cargo test --lib m2c_probe_alive_device_sram_struct -- --nocapture

natural boot: n=53659 stop=Unknown pc=0x8cb1 word=0x61a800
firmware emitted no device-SRAM descriptor stores
```

That failure is the pre-existing unresolved production-map collision. The
candidate-specific million-instruction run separately retains both positive
landmarks (`publisher_pass`, `service_entered`) while proving that the
dissolving candidate still does not publish. The execution-guided probe exits
101 by design for this result: it reports zero solutions and one inconclusive
budget candidate, rather than turning a budget exhaustion into a false pass.

## Retained evidence

The exploratory instrumentation was removed after capture; it did not change
firmware, CPU, MMU, device, or scheduler state. Logs remain under
`build/experiments/firmware-re/`:

- `c961-spin-trace.log` -- pre-fix 32-pass `LCOUNT` trace and aggregate counts;
- `c961-hybrid-lowstores.log` -- delta-zero publisher writes and the `0x94ea`
  overwrite;
- `c961-uniform-at-lowstores.log` -- coherent-AT control;
- `c961-runtime-base-popcount.log` -- coherent-publisher/BASE-service
  counterfactual returning 1 and advancing to `0x7fe7`;
- `c961-post-excm-fix.log` -- post-fix service-tail hits proving the
  145-instruction outer recurrence;
- `delta-split-search.tsv` -- post-fix million-instruction candidate row and
  final instruction tail, including the single pass through each helper loop.

Static/runtime reproducers already in tree:

The first command is expected to exit 101 with the candidate classified as
inconclusive (`service:budget`); it is an evidence probe, not a passing
regression.

```bash
XDNA_FW_PROBE=1 XDNA_FW_ONLY=0x100:0:0x8cae:0x5c \
  XDNA_FW_MAX=1000000 XDNA_FW_JOBS=1 XDNA_FW_TRACE=1 \
  cargo test --lib m2c_probe_execution_guided_framing_search -- --nocapture

XDNA_FW_PROBE=1 XDNA_FW_MAX=100000 \
  cargo test --lib m2c_probe_runtime_view_discriminator -- --nocapture

XDNA_FW_PROBE=1 XDNA_FW_DISASM=c938:c980 XDNA_FW_DISASM_FILEOFF=5c \
  cargo test --lib m2c_probe_disasm_range -- --nocapture
```

No production mapping or firmware state was changed. The only production
change is the faithful `PS.EXCM` gate on zero-overhead-loop back-edges and its
fill-fastpath equivalent.
