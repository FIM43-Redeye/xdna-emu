# Go-alive dispatch target and completion

**Date:** 2026-07-11  
**Target:** Phoenix/NPU1 AIE2 management firmware `1502_00/npu.dev.sbin`  
**Firmware SHA-256:** `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`  
**Status:** Complete. Execution refutes the premise that this `waiti` awaits a
hardware completion.

## Questions

1. What completion is `goalive_runfn` waiting for at the `waiti` whose resume
   PC is VMA `0x5645`, and what firmware writes arm that source?
2. When Xtensa interrupt line 0 is pending, what software table slot does the
   shared level-1 vector path read, and what handler and argument are stored in
   that slot by the natural boot?

## Evidence rules

- A natural-boot executed store or read is primary evidence. Each behavioral
  claim must retain the instruction count, PC, effective address, and value
  where the probe can expose them.
- Static code is admissible only for the post-vector path that natural boot
  does not execute. Every static step must state its VMA, file offset/framing,
  decoded instruction, and data source.
- A clean decode is not sufficient to select between the BASE (`file=VMA+0x5c`)
  and AT (`file=VMA+0x100`) views.
- The injected `0x7fc4 -> 0x8c6c` path is excluded as a dispatch oracle. The
  committed natural-boot reachability probe found no execution of `0x7fc4`,
  `0x7fe1`, `0x8c6c`, or `0x2958` before the go-alive wait.
- This is a localization report. No emulator fix is part of this change.

## Reproducible baseline

At commit `74cc390e`, `FirmwareProcessor::load_m2c` plus
`enable_host_mailbox()` and `boot_to_idle(n)` reaches the staged-channel
milestone: `local_data[0x14820] == 0x5550_4e5f` (`"_NPU"`) and parks with
resume PC `0x5645`, `INTENABLE=1`, `PS.INTLEVEL=0`, and `PS.EXCM=0`.

Prior execution established only this architectural fact: making Xtensa line
0 pending wakes any `waiti 0`. It did not establish a causal device source.
The prior ZDMA identification was an offset coincidence, not a Phoenix
absolute-address assignment. The executed helper treats `0x27200904..` as an
array of packed two-bit fields, which is incompatible with interpreting those
same words as ZDMA `CH_IMR`, `CH_IEN`, and status registers.

## Step 1: natural-boot ISR installation

### Result: two runtime tables, not a Zephyr `{isr,arg}` array

A temporary observation probe booted naturally with the same `load_m2c` plus
column-ready agent as the milestone, decoded every executed 32-bit store, and
recorded code-valued writes before the wait. It reached `n=52390`, PC
`0x5645`. The probe was removed after capture; no emulator behavior changed.

The boot constructs two related tables:

- **Dispatch records:** base VMA `0x110b0`, stride `0x14`. Each record has the
  hardware IRQ/source ID at `+0`, handler at `+0xc`, and argument at `+0x10`.
- **IRQ-to-record map:** one byte per hardware IRQ at VMA `0x11700 + irq`.
  The byte is the record index.

The natural boot installed 58 non-empty records. The executed groups are:

| hardware IRQ IDs | record slots | handler VMA | argument |
|---|---:|---:|---:|
| `0x20..0x2f` | `0..15` | `0x5948` | `0..15` |
| `0x38..0x3b` | `16..19` | `0x59f0` | `0..3` |
| `0x4c..0x4f` | `20..23` | `0x5974` | `0..3` |
| `0x37` | `24` | `0x5a20` | `0` |
| `0x36` | `25` | `0x5a3c` | `0` |
| `0x60..0x7f` | `26..57` | `0x907c` | the IRQ ID itself |

For example, the handler stores for IRQs `0x20` and `0x21` execute at
`n=41582` and `n=41687`, both at PC `0x898d`, to EAs `0x110bc` and
`0x110d0`, with value `0x5948`. The corresponding final records are:

```text
0x110b0: irq=0x20, handler@0x110bc=0x5948, arg@0x110c0=0
0x110c4: irq=0x21, handler@0x110d0=0x5948, arg@0x110d4=1
```

The same executed install site writes `0x5974` to `0x1124c`, `0x11260`,
`0x11274`, and `0x11288` for IRQs `0x4c..0x4f` at
`n=43568,43665,43762,43859`. The final record and byte-map contents were then
read directly from that natural boot state.

This is not Zephyr's usual two-word `_sw_isr_table` representation. It is a
firmware/MERT interrupt-controller registry with explicit IRQ metadata and a
separate dense lookup map. No `_sw_isr_table` symbol is present in the stripped
`build/experiments/firmware-re/symbols.txt`.

## Step 2: shared level-1 dispatch

### Result: line 0 is an aggregate input; there is no unique “line-0 slot”

The static chain is coherent only with the following framing:

| VMA | file offset | instruction / effect |
|---:|---:|---|
| `0xae0` | `0xb3c` (BASE `+0x5c`) | `wsr EXCSAVE1,a3` |
| `0xae3` | `0xb3f` (BASE) | `l32r a3,[0xadc]`; literal is `0x28b4` |
| `0xae6` | `0xb42` (BASE) | `jx a3` to `0x28b4` |
| `0x28c3` | `0x29c3` (AT `+0x100`) | `rsr a3,EXCCAUSE` (SR `0xe8`) |
| `0x28dc` | `0x29dc` (AT) | distinguish syscall cause `1` from the interrupt arm |
| `0x29c2` | `0x2ac2` (AT) | reload saved cause from `EXCSAVE3` |
| `0x29da` | `0x2ada` (AT) | load literal VMA `0x2898` = pointer `0xd864` |
| `0x29dd` | `0x2add` (AT) | `callx4 a4` to `0xd864` |
| `0xd864` | `0xd964` (AT) | clean `ENTRY`; BASE is mid-function bytes |
| `0xd885` | `0xd985` (AT) | cause `4` branches to `0xd8bc` |
| `0xd8bc` | `0xd9bc` (AT) | `call8 0x8784` |
| `0x8784` | `0x8884` (AT) | interrupt-controller dispatcher entry |

This exposes the immediate cause of the phantom injection path. Production
`load_m2c` maps AT bytes only from `0xd8a7` onward in this section; it does not
cover the real AT entry at `0xd864`, nor the AT dispatcher at `0x8784`.
Consequently the current injected interrupt calls the correct static pointer
`0xd864` but fetches unrelated BASE bytes there, which eventually manufactures
the `0x7fc4 -> 0x8c6c` path. The handler pointer computation is firmware code;
the emulator's missing input is the correct code framing and interrupt-controller
state, not a replacement hardcoded jump.

The correctly framed `0x8784` dispatcher does the following:

1. `0x8787` loads literal VMA `0x3484` = MMIO `0x272003c4`;
   `0x878d` reads it and `0x878f` extracts its low byte as the current hardware
   IRQ/source ID.
2. IDs `0x4c..0x50` take a special aggregate-status arm using MMIO
   `0x272003b8`; the ordinary arm begins at `0x87cc`.
3. The ordinary arm uses saved mask words at `0x116f0` and the controller mask
   registers beginning at `0x27200300`.
4. `0x87f2..0x87f4` reads byte `*(0x11700 + irq)` to obtain the dispatch-record
   slot. Literal VMA `0x3490` supplies record base `0x110b0`.
5. Two `addx4` instructions multiply the slot by `20`; `0x8800` loads the
   handler from `record+0xc`, `0x8802` loads the argument from `record+0x10`,
   and `0x8804` executes `callx8 handler(arg)`.

The software path does **not** read Xtensa `INTERRUPT` SR `0xe2` to identify
the source. Hardware has already used `INTERRUPT & INTENABLE` to vector the CPU;
software then reads the on-chip controller's source-ID MMIO at `0x272003c4`.
The interpreter models SRs `INTERRUPT=0xe2`, `INTCLEAR=0xe3`, and
`INTENABLE=0xe4` at `src/firmware/xtensa/interp/mod.rs:75-86`, and its delivery
gate at `interp/mod.rs:489-498`, but it has no model for the controller source ID
or lookup-side MMIO.

Therefore Q2's requested singular line-0 handler does not exist. Xtensa bit 0
is the aggregate level-1 input; the real target depends on the source ID read
from `0x272003c4`. Merely setting `cpu.interrupt |= 1` leaves that register at
zero. Since `0x11700[0]` is also zero-initialized, an otherwise correctly framed
run would accidentally select record slot 0—IRQ `0x20`, handler `0x5948`,
argument `0`—which is not evidence that IRQ `0x20` actually fired.

### Source-specific slots relevant to go-alive

The go-alive setup enables controller sources `0x2e` and `0x2d`. They resolve
as follows; these are controller-source records, not CPU-line records:

| source ID | map byte | record VMA | handler field | handler | arg field | arg |
|---:|---:|---:|---:|---:|---:|---:|
| `0x2d` | `[0x1172d]=13` | `0x111b4` | `0x111c0` | `0x5948` | `0x111c4` | `0x0d` |
| `0x2e` | `[0x1172e]=14` | `0x111c8` | `0x111d4` | `0x5948` | `0x111d8` | `0x0e` |

Handler `0x5948` is AT-framed at file offset `0x5a48`. Its static body indexes
an eight-byte per-channel metadata table, posts a MERT event, clears controller
source `arg+0x20`, writes `1` to Xtensa `INTCLEAR` (SR `0xe3`) to deassert the
aggregate CPU input, and re-enables the controller source. The natural state at
the first wait has zero metadata for channels 13 and 14; no executed store
initialized those two entries before the wait. That is another reason not to
promote either source to “the completion” without observing it fire.

A probe-only correctly framed dispatch sweep seeded `0x272003c4` with source
IDs `0x20`, `0x25`, `0x26`, `0x4c`, `0x6a..0x6d`, and `0x70`, then asserted
CPU bit 0. Every arm reached `0xd864`, `0x8784`, and the handler selected by the
runtime table: `0x5948` for `0x20/25/26`, `0x5974` for `0x4c`, and `0x907c`
for `0x6a..6d/70`. No arm wrote `FW_ALIVE_OFF`; `0x907c` exposed a further
framing cell at `0x907e`. The sweep was localization evidence only and was
removed afterward.

## Step 3: go-alive arming and completion

### Result: the `waiti` is caused by one missing literal overlay

The decisive instruction is not `waiti`; it is the guard that precedes it:

| VMA | file offset / framing | instruction | observed effect |
|---:|---:|---|---|
| `0x562b` | `0x572b` (AT) | `l32r a7,[0x31a4]` | should load `0x27010ac0` |
| `0x563e` | `0x573e` (AT) | `l32i.n a2,[a7]` | read the status word |
| `0x5640` | `0x5740` (AT) | `beqz.n a2,0x5645` | zero skips the wait |
| `0x5642` | `0x5742` (AT) | `waiti 0` | only executes for nonzero status |

The literal at VMA `0x31a4` has two different raw views:

```text
BASE file 0x3200 (VMA+0x5c): 0x00000000
AT   file 0x32a4 (VMA+0x100): 0x27010ac0
```

The iter25 overlay list includes the adjacent `0x31ac..0x31b0` literal but
omits `0x31a4` (`src/firmware/mod.rs:190-225`). Because `fetch8` chooses an
overlay by VMA (`src/firmware/mmio.rs:200-226`), the natural emulated boot
loads BASE `0` into `a7`. The executed trace is therefore:

```text
n=52380 pc=0x562b  l32r a7,[0x31a4]          -> a7=0
n=52388 pc=0x563e  l32i.n a2,[a7] EA=0       -> 0x030bb000
n=52389 pc=0x5640  beqz.n a2,0x5645           not taken
n=52390 pc=0x5642  waiti 0                    parks, resume PC 0x5645
```

`0x030bb000` is only the value accidentally found in local word zero. It is
not the status that this code tests.

The correct status word is live elsewhere in the same natural boot. The
peripheral-access capture records a read of `0` from `0x27010ac0` at high VMA
`0x20003dcf`, followed by a write of `0` at `0x20003dd4`; its value at the
alleged gate is still zero
(`build/experiments/firmware-re/m2c_probe_waiti_wake_condition_20260711.log:5-12,165`).
Thus the real branch condition is already satisfied: firmware should skip
`waiti`.

Execution confirms it. A probe-only `add_rom_overlay(0x31a4,0x31a8,0x100)`
changes no data or peripheral state. On the next natural boot:

```text
n=52380 pc=0x562b  l32r                  -> a7=0x27010ac0
n=52388 pc=0x563e  l32i.n EA=0x27010ac0 -> 0
n=52389 pc=0x5640  beqz.n                taken to 0x5645
n=52390 pc=0x5645  movi a10,-1           (no interrupt was injected)
```

The next instruction sequence exposed two more previously unreachable framing
gaps, rather than a missing completion:

1. Literal VMA `0x32c8` is BASE `0x00015ff0` but AT `0x08b04428`. Adding the
   coherent AT pool `0x32c8..0x32d4` makes `0x564d` call the actual MERT
   event-get primitive. It executes the syscall/shared-exception path, returns
   event class `0`, and resumes at `0x5650`, still without an injected
   interrupt.
2. AT-framing the called helper at `0x7c5c` advances a further 236 instructions
   through event processing and scheduler code before the next independent
   decode gap at `0x26d4`. `FW_ALIVE_OFF` remains zero at that later frontier.

Those latter gaps belong to the mapping arc that the primary will implement;
they do not restore a completion interpretation for `0x5642`.

### What go-alive actually arms

There are two interrupt layers:

- **CPU aggregate line 0:** armed once during global interrupt-controller init,
  not immediately before the go-alive wait. At `n=2217`, PC `0x88d5` executes
  `wsr INTENABLE,a2` with value `1`. The surrounding code snapshots controller
  mask words `0x27200300..0x2720030c` into `0x116f0..0x116fc`, then ORs bit 0
  into Xtensa `INTENABLE`.
- **Controller sources:** go-alive enables source `0x2e` at `n=51975` and
  source `0x2d` at `n=52085`. Both stores execute at PC `0x871c` and write the
  accumulated mask `0x0fc06000` to `0x27200304`. These are the source-specific
  records shown above, both dispatched to `0x5948` with arguments `0x0e` and
  `0x0d` respectively.

Go-alive also executes packed two-bit field updates for selectors
`0x70,0x15,0x17,0x16,0x0e,0x26,0x6c,0x6d,0x0d,0x25,0x6a,0x6b` through the
helper at `0x8d88..0x8db2`. The exact formula is:

```text
word  = 0x27200904 + 4 * (selector >> 4)
shift = 2 * (selector & 0xf)
word  = (word & ~(3 << shift)) | (value << shift)
```

That is interrupt-routing/configuration state, not a DMA descriptor or kick.
The executed go-alive path writes no source/destination/length descriptor and
does not read a DMA completion status. Consequently the earlier proposed
`0x27200800` ZDMA `DMA_DONE=0x400` completion is refuted for this gate.

### What “raises line 0” on real hardware

There is no MMIO address to which “real hardware writes line 0.” A peripheral
asserts one of the interrupt controller's source inputs; the controller exposes
the selected source ID at `0x272003c4` and asserts the aggregate Xtensa input.
The CPU then observes bit 0 in SR `INTERRUPT` (`0xe2`). Firmware clears the
aggregate with `wsr INTCLEAR,1` (SR `0xe3`) after handling the controller source.

For faithful synthetic delivery of a *particular* interrupt, the emulator must
therefore model both levels: present the source ID/status/mask in the
`0x272003xx` controller and assert CPU pending bit 0. Setting only
`cpu.interrupt |= 1` is under-specified. But no such delivery is required to
pass the branch at `0x5640`; the correct status value is zero and the firmware
does not execute `waiti` there.

## Final answers

### Q1 — completion

**None.** The observed `waiti` is an emulator mapping artifact. Its guard is
supposed to read `0x27010ac0` through the AT literal at VMA `0x31a4`; that word
is zero, so real execution branches over `waiti`. The current map serves the
BASE literal `0`, reads unrelated local word zero (`0x030bb000`), sees a
nonzero value, and sleeps. `0x272009xx` is packed two-bit controller
configuration, not a go-alive ZDMA completion block, and there is no exact
device write to synthesize for this gate.

### Q2 — dispatch target

Xtensa line 0 has no singular handler slot. It is the aggregate input. The
shared vector chain is:

```text
VECBASE+0x2e0 (0xae0 BASE) -> 0x28b4 AT -> cause-4 arm at 0xd864 AT
-> controller dispatcher 0x8784 AT -> read source ID [0x272003c4]
-> slot = *(u8 *)(0x11700 + source)
-> record = 0x110b0 + slot*0x14
-> callx8 record.handler(record.arg)
```

For the two controller sources enabled by go-alive, source `0x2d` selects
record `0x111b4`, handler `0x5948`, argument `0x0d`; source `0x2e` selects
record `0x111c8`, handler `0x5948`, argument `0x0e`. The CPU-facing SRs are
`INTERRUPT=0xe2`, `INTCLEAR=0xe3`, and `INTENABLE=0xe4`
(`src/firmware/xtensa/interp/mod.rs:75-86`). The source-selecting MMIO is
`0x272003c4`; source masks begin at `0x27200300`; the special `0x4c..0x50`
status arm reads `0x272003b8`.

The implementation consequence is mapping-first: add the proven AT literal
and newly reached post-guard ranges, then continue execution. Do not synthesize
a ZDMA completion or hardcode a “line-0 handler.”
