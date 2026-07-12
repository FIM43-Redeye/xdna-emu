# The `0x8c6c` service path is real; `EXCCAUSE=1` was stale state

Date: 2026-07-11  
Target: Phoenix/NPU1 firmware `1502_00/npu.dev.sbin`  
SHA-256: `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`  
Branch: `feat/m2c-mapping-boot-to-idle`

## Verdict: WIN B

The naturally executed `0x8770 -> 0xc530 -> 0x7fc4 -> 0x8c6c` service chain is
genuine and BASE-framed. Its root is a live, uniquely written callback pointer
loaded from a BASE-framed literal whose AT value is a different known handler.
Every downstream target is then produced by an executed direct or indirect
call. `0x8c6c` is not the result of an exception table, a guessed symbol, or a
coherent-only static descent.

The brief's syscall premise was wrong in one precise way: `0x7fe1` is a
`Call8 0x8c6c`, not a `SYSCALL`, and the exception handler does not dispatch to
it. All five natural syscalls occur earlier. `EXCCAUSE=1` at `0x7fe1` is stale
architectural state from the last syscall, not evidence that the current
control transfer is an exception arm.

This correction does not dissolve the collision. The publisher remains pinned
to AT and the service remains pinned to BASE, so both genuinely require
different file bytes at identity-mapped PA `0x8cae`. The standard Xtensa
selectors already tested (ITLB remap and runtime memory-copy overlay) remain
eliminated. PSP load-time behavior or an instruction-fetch mechanism below the
modeled MMU is therefore the pinned next investigation; no `load_m2c` boundary
change is justified.

## 1. The syscall path and service path are separate

`m2c_probe_service_path_provenance` records every executed `Syscall` before the
frontier:

```text
n=47433 pc=0x08b043e1
n=49543 pc=0x08b04262
n=52222 pc=0x08b04444
n=52654 pc=0x08b04279
n=52998 pc=0x08b04262
```

Each syscall immediately takes the real BASE vector stub at `0x0ae0`, which
loads/jumps to the AT-framed unified handler at `0x28b4`. For the last syscall:

```text
n=52998 pc=0x08b04262 Syscall
n=52999 pc=0x00000ae0 Wsr EXCSAVE1       EXCCAUSE=1 EPC1=0x08b04262
n=53002 pc=0x000028b4 Wsr EXCSAVE2       EXCCAUSE=1 EPC1=0x08b04262
```

The service call happens 641 instructions later through ordinary calls:

```text
n=53567 pc=0x0000283b Callx8 a5          a5=0x8770
n=53568 pc=0x00008770 Slli
n=53578 pc=0x0000878a Call8 0xc530
n=53579 pc=0x0000c530 Entry
n=53630 pc=0x0000c56e Call8 0x7fc4
n=53631 pc=0x00007fc4 Entry
n=53639 pc=0x00007fe1 Call8 0x8c6c       EXCCAUSE=1 EPC1=0x08b04265
n=53640 pc=0x00008c6c Entry
```

`EPC1=0x08b04265` is exactly the last syscall PC plus three, as advanced by the
firmware handler. The `Entry` at `0x8c6c` later raises a register-window
overflow, so the second execution of `0x8c72` has `EPC1=0x8c72`; neither value
makes `0x7fe1` an exception dispatch. The executed opcode at `0x7fe1` is
unambiguously `Call8 { target: 0x8c6c }` (`a5 c8 00`).

## 2. The BASE service root is pinned by a live two-view discriminator

Initialization reads the callback from VMA `0x32f4`:

```text
n=41484 pc=0x200046c4 L32r a10,[0x200032f4]
VMA 0x32f4 BASE file 0x3350 = 0x00008770
VMA 0x32f4 AT   file 0x33f4 = 0x00005948
n=41488 pc=0x2000dae2 [0x1187c] <- 0x00008770
```

`0x5948` is the separately identified interrupt-record handler, not a shifted
alias of the service callback. The firmware writes `[0x1187c]` exactly once.
At the later scheduler dispatch, the same live value is consumed:

```text
n=53562 pc=0x00002830 L32iN a6,[a9+16] EA=0x1187c -> a6=0x8770
n=53567 pc=0x0000283b Callx8 a5          a5=0x8770
```

That executed callback directly reaches the BASE descriptor builder at
`0xc530`, which directly reaches the BASE caller at `0x7fc4`, whose encoded
PC-relative call reaches `0x8c6c`. The alternative views do not provide the
same entries:

| VMA | BASE (`+0x5c`) | AT (`+0x100`) |
|---|---|---|
| `0xc530` | `36 61 00` = `Entry a1,0x30` | body of a different function |
| `0x7fc4` | `36 41 00` = `Entry a1,0x20` | body of a different function |
| `0x8c6c` | `36 41 00` = `Entry a1,0x20` | `22 2a 00` = `L32i` |

Thus the service framing is anchored by an executed literal value, a unique RAM
registration, a later RAM read, an indirect call, and three direct call edges.
It is stronger evidence than decode coherence alone.

## 3. The publisher remains independently pinned to AT

The publisher root is also an executed absolute pointer, not a target selected
by the service analysis:

```text
n=47328 pc=0x003dd9 L32r a10,[0x2000324c]
VMA 0x324c BASE file 0x32a8 = 0x000055f8
n=47335 pc=0x003de9 Call8 task_create
n=47336 pc=0x00d664 task_create Entry     a2=0x55f8
n=47362 pc=0x00d6e6 [0x2320] <- 0x55f8
```

At VMA `0x55f8`, only AT supplies the required `Entry` (`36 81 00`); BASE
supplies `e6 11 7a`. Natural execution then follows AT-framed direct calls
`0x55fb -> 0x50d4`, `0x50f1 -> 0x8f44`, and `0x9045 -> 0x8c98`. The function
at `0x8c98` executes five times and reaches the real `waiti` at `0x5645`.

The BASE aliases `0x5178 -> 0x8fe8 -> 0x8d3c` are statically coherent, but they
are unreachable from the live absolute root `0x55f8`. Replacing the shared
`0x8cae..0x8cbc` cell with BASE breaks the executed publisher at `0x8c32`
before `_NPU` is built.

## 4. No single static physical assignment serves both paths

The executed service takes `Bbci @0x8c8b -> 0x8cae`. Under BASE the tail is:

```text
0x8cae Addi  a8,a8,0x60
0x8cb1 Addmi a4,a4,0x1000
0x8cb4 Addmi a5,a5,0x1000
0x8cb7 Wsr   PS,a3
0x8cba RetwN
```

The executed AT publisher's three-byte `Bgeu @0x8cac` occupies the same
physical byte at `0x8cae`, then uses AT `MoviN @0x8cb4`. Fork A established
that both fetch VMA `0x8cae` as identity-mapped PA `0x8cae`, with byte-identical
ITLB state and no ITLB operation touching the page.

The four static assignments remain exhaustive:

```text
code=BASE literal=BASE: publisher walls at 0x8c32, magic=0
code=BASE literal=AT:   publisher walls at 0x8c32, magic=0
code=AT   literal=BASE: publisher reaches 0x5645; service walls at 0x8cb1
code=AT   literal=AT:   publisher reaches 0x5645; service walls at 0x8cb1
free section variables: []
```

There is therefore no byte-justified `load_m2c` tuple or boundary correction to
land in this pass.

## 5. `a7=6` is a real later reject, not the service-entry discriminator

The test-only runtime-view counterfactual lets the first BASE service return,
then selects BASE for the encoded `0x7fe7 -> 0x26d4` alias. The later path
loads the rejected value from task state:

```text
n=53804 pc=0x2720 L32iN a8,[a7+40]   EA=0x2278 -> 0x10dfc
n=53807 pc=0x2728 L32iN a15,[a8+8]   EA=0x10e04 -> 6
n=53813 pc=0x2734 Call8 0xc530        a10..a15={1,0x15,0,0,0x7fea,6}
n=53815 pc=0xc533 Entry/Rsil          callee a7=6
n=53871 pc=0xc56e Call8 0x7fc4
n=53873 pc=0x7fc7 Bgeui a7,6,0x7fec   taken
n=53874 pc=0x7fec J 0x7fec
```

The value is not created by the `0x8cxx` misdecode: coherent code loads it from
`current_task + 8`, passes it through the windowed ABI, and the caller's explicit
unsigned bounds check rejects it. This is a legitimate reject of a second IPC
message reached only after the counterfactual crosses the original collision.
It says that the two test-selected view changes are insufficient to model the
hardware path; it does not make the earlier `a7=0` call to `0x8c6c` an artifact.

## Consequence

This pass refutes reconstruction-error hypothesis (i) for the `0x8c6c` service
root and framing. It does not identify the missing hardware mechanism. Given the
separate negative results for runtime copying and ITLB selection, the next
bounded step is PSP-loader RE: determine whether PSP load metadata duplicates,
patches, or banks the overlapping low-VMA bytes before considering an exotic
instruction-fetch feature.

No production overlay, MMU, scheduler, interrupt, or firmware state was
changed. The only code changes are additive `XDNA_FW_PROBE`-gated observations.

## Reproduction

```text
XDNA_FW_PROBE=1 cargo test --lib \
  m2c_probe_service_path_provenance -- --nocapture

XDNA_FW_PROBE=1 cargo test --lib \
  m2c_probe_execution_guided_framing_search -- --nocapture

XDNA_FW_PROBE=1 XDNA_FW_MAX=100000 cargo test --lib \
  m2c_probe_runtime_view_discriminator -- --nocapture

XDNA_FW_PROBE=1 cargo test --lib \
  m2c_probe_itlb_code_view_selector -- --nocapture
```
