# Firmware go-alive async-completion characterization

Date: 2026-07-11  
Target: Phoenix/NPU1 AIE2 management firmware `1502_00/npu.dev.sbin`
SHA-256: `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`

## Executive verdict

The premise that `FUN_00008d98` loads a copy descriptor and kicks an engine is
refuted by executed instructions. The actual helper begins at VMA `0x8d88`,
file offset `0x8e88` (`+0x100` framing), and performs a generated-looking
two-bit field update:

```text
word = 0x27200904 + 4 * (selector >> 4)
shift = 2 * (selector & 0xf)
new = (old & ~(3 << shift)) | (value << shift)
```

The strongest open-source identification for the aperture is a Xilinx
ZDMA/ADMA channel: the offsets from an inferred core base `0x27200800` match
the ZDMA register map exactly. This is a high-confidence IP-family
identification, not a proved Phoenix absolute-address assignment: xdna-driver,
aie-rt, and the AM025 register database contain no naming for `0x272009xx`.

The injected line-0 trace does not read an engine completion status.
`0xd830` is `LSI ft10,[a12+0x18c]`, part of context save/restore; varying the
backing value at `0x0405078c` across `0`, `1`, `0x400`, `0x800`, `0xc00`,
`0xffff`, and `0xffffffff` leaves the path identical. The only minimal wake
condition proved for `goalive_runfn` is a deliverable Xtensa interrupt
line 0. No firmware-side ZDMA status/ack access was observed on that path, so
the causal routing from this IP block to line 0 remains unresolved.

Finally, the `0x8cb1` decoder wall is a framing artifact. The publish helper at
VMA `0x8c98` genuinely executes from `file=VMA+0x100`; the separate BASE-framed
service routine entered at `0x8c6c` branches into a tail at `0x8cae..0x8cba`
whose coherent bytes are `file=VMA+0x5c`. The current vaddr-only overlay
`[0x8c98,0x8d52)` cannot represent both legitimate uses and over-claims the
BASE tail.

## 1. What the `0x272009xx` writes actually are

### Executed helper and framing

The symbol-map name `FUN_00008d98` points inside the executed function. A
live-bus decode starting at that label therefore immediately reports Unknown
(`build/experiments/firmware-re/re_probe_fun_8d98_body_and_callers.log:5-23`).
Execution enters at VMA `0x8d88`, file `0x8e88`, and runs through the return at
VMA `0x8db2`, file `0x8eb2`:

| VMA | File | Instruction | Role |
|---:|---:|---|---|
| `0x8d88` | `0x8e88` | `ENTRY` | function entry |
| `0x8d8b` | `0x8e8b` | `L32R a4,0x34e4` | literal is `0x27200904` |
| `0x8d8e` | `0x8e8e` | `SRLI a6,a2,4` | register-word index |
| `0x8d91` | `0x8e91` | `EXTUI a15,a2,0,4` | field index within word |
| `0x8d94` | `0x8e94` | `ADDX4 a4,a6,a4` | select word |
| `0x8d97` | `0x8e97` | `SLLI a2,a15,1` | two-bit shift |
| `0x8d9a` | `0x8e9a` | `L32I.N a6,[a4]` | read old word |
| `0x8d9c..0x8dad` | `0x8e9c..0x8ead` | mask/merge | replace one two-bit field |
| `0x8db0` | `0x8eb0` | `S32I.N a2,[a4]` | write merged word |
| `0x8db2` | `0x8eb2` | `RETW.N` | return |

The decisive final invocation is
`0x5038 MOVI a10,13; 0x503a MOVI a11,3; 0x503c CALL8 0x4a78`, forwarded by
`0x4a7b..0x4a7f` to `0x8d88`. At `n=52109`, VMA `0x8db0` merges value `3`
at shift 26 into old `0x30000000`, producing `0x3c000000` at MMIO
`0x27200904` (`exec_trace_goalive_50000_52400.log:2094-2116`). Thus
`0x3c000000` is register-field data, not an address supplied as a destination.

The alleged kick is likewise a field update. At VMA `0x5175` the caller passes
selector `0x70`, value `1`; that selects `0x27200920`, shift zero, and writes
`1` at `n=50974` (`exec_trace_goalive_50000_52400.log:956-981`). It occurs more
than a thousand executed instructions before the final `0x3c000000` update.

### Register-field verdict

The five flat-stub shadow values at the gate are real observed writes
(`m2c_probe_waiti_wake_condition_20260711.log:42-49`), but they are not a
descriptor:

| Address | Shadow value | Correct characterization |
|---:|---:|---|
| `0x27200904` | `0x3c000000` | packed two-bit field updates; not destination |
| `0x27200908` | `0x0000e400` | packed two-bit field updates; not byte size |
| `0x2720090c` | `0x00003c00` | packed two-bit field updates |
| `0x2720091c` | `0x0ff00000` | packed two-bit field updates |
| `0x27200920` | `0x00000001` | packed two-bit field update; not a kick |

The actual register-family cross-reference below makes the descriptor claim
structurally impossible too: source descriptors begin at channel offset
`+0x128`, destination descriptors at `+0x138`, and total-byte status is at
`+0x188` (`xzdma_hw.h:46-83`). The gate trace contains no access to inferred
absolute addresses `0x27200928..0x27200944`.

### No relationship to `mgmt_mbox_chann_info`

The final `0x3c000000` write occurs at `n=52109`. Only afterward, at `n=52172`,
VMA `0x504a` loads literal `0x14800` and starts building the channel structure;
the `_NPU` magic is stored at `0x14820` at `n=52223`
(`exec_trace_goalive_50000_52400.log:2115-2116,2178-2187,2217-2230`). Later
address-bearing encodings write `0x01f94800` to `0x272100f8`, `0x1ff94400` to
`0x272100fc`, and `0x1ff94000` to `0x272100bc`
(`exec_trace_goalive_50000_52400.log:2230-2246,2267-2289`). Their public
register names and transfer semantics are not established here.

The driver defines the 64-byte channel-info layout in
`xdna-driver/src/driver/amdxdna/aie2_pci.c:51-79` and says firmware publishes
it in the SRAM BAR, then writes its address at `FW_ALIVE_OFF`
(`aie2_pci.c:202-250`). NPU1's SRAM device base is `0x03080000`, mailbox base
is `0x030c0000`, and `FW_ALIVE_OFF` resolves to `0x030bf000`
(`npu1_regs.c:22-41,72-89`). `0x3c000000` is therefore not a host-visible SRAM
address in the driver map. It happens to be the emulator's synthetic page-table
base (`src/firmware/mmio.rs:82-86`); that numerical coincidence does not make
the MMIO field value a pointer.

## 2. Engine identification and generic completion semantics

The open-source Xilinx ZDMA register map is an exact offset match if the
Phoenix channel core base is `0x27200800`:

| Absolute | ZDMA offset/name |
|---:|---|
| `0x27200900` | `+0x100 CH_ISR` |
| `0x27200904` | `+0x104 CH_IMR` |
| `0x27200908` | `+0x108 CH_IEN` |
| `0x2720090c` | `+0x10c CH_IDS` |
| `0x27200910` | `+0x110 CH_CTRL0` |
| `0x27200914` | `+0x114 CH_CTRL1` |
| `0x27200918` | `+0x118 CH_PERIF` |
| `0x2720091c` | `+0x11c CH_STS` |
| `0x27200920` | `+0x120 CH_DATA_ATTR` |

Source: `amd-unified-software/embeddedsw/.../zdma_v1_19/src/xzdma_hw.h:41-63`.
This identifies the likely IP family as the management processor's AXI
ZDMA/ADMA, not AIE array-tile BD/DMA and not an SRAM-specialized copy engine.
The absolute assignment remains an inference because the searched xdna-driver,
aie-rt, and AM025 sources do not name this management-processor block.

Generic ZDMA completion is:

1. `CH_ISR.DMA_DONE` is bit `0x400`; `CH_ISR` supports W1C, and `CH_IMR` uses
   zero for enabled and one for disabled (`xzdma_hw.h:87-122`,
   `xzdma.h:369-433`).
2. The reference handler computes `pending = ISR & ~IMR`, invokes the done
   callback on `pending & 0x400`, disables interrupts, and W1C-clears pending
   ISR bits (`xzdma_intr.c:70-110`).
3. Platform code is responsible for connecting that handler to an interrupt
   system (`xzdma_intr.c:48-61`). Nothing in that source says Phoenix maps it
   to Xtensa line 0.

The flat firmware stub records action/RO-style registers as ordinary mutable
words, so its reported “final state” must not be interpreted as real silicon
readback. In particular, treating the `+0x104` shadow as a writable IMR state
or the `+0x11c` shadow as a control word would exceed the evidence.

## 3. Observed ISR and the minimal completion contract

At the gate, the proved CPU state is `INTENABLE=1`, `INTERRUPT=0`,
`intlevel=0`, `EXCM=false` (`m2c_probe_waiti_wake_condition_20260711.log:5-8`).
`goalive_runfn` executes `WAITI 0` at VMA `0x5642`, file `0x5742`; its resume
PC is VMA `0x5645`, file `0x5745`, with no immediately following MMIO test
(`disasm_goalive_overlay.log:31-40`).

After artificial line-0 assertion, the trace reaches VMA `0xd830` (BASE file
`0xd88c`) and decodes `LSI ft10,[a12+0x18c]`, then VMA `0xd833` stores a byte
and VMA `0xd836` calls the scheduler (`re_probe_goalive_completion_values.log:8-24`).
Changing the array backing at effective address `0x0405078c` through seven
candidate status values produces the same instruction path and same
`0x8cb1` stop (`re_probe_goalive_completion_values.log:8-134` and subsequent
blocks). Therefore that load is floating context state, not completion status,
and its value does not control dispatch.

No `0x27200900` status or ack access occurs after injection; the only observed
post-injection peripheral read is the context `LSI` backing read and the only
new store is the context byte at `0x40002a0c`
(`m2c_probe_waiti_wake_condition_20260711.log:1091-1139`). A whole-image
literal scan finds only four `0x27200904` base loads and no ISR-base reference
(`literal_xref_27200800_27200a00.log:5-10`).

Accordingly:

- **Minimal wake condition proved for go-alive:** assert Xtensa line 0
  while bit 0 is enabled, `intlevel=0`, and `EXCM=false`.
- **Minimal faithful device state:** unresolved. If this aperture is ZDMA, a
  real device completion would normally set `CH_ISR.DONE=0x400` and assert its
  platform IRQ, but the observed firmware line-0 path neither tests nor clears
  it. The firmware may rely on another interrupt layer, a callback/task already
  made runnable, or hardware auto-routing not visible in this trace.
- **No observed “event post” in this handler:** an older correctly BASE-framed
  line-0 run entered the service callback and returned by `RFE` in 282 steps
  while the observed event accumulator at `0x22bc` was unchanged
  (`isr_observe.log:3-18`). The current evidence supports direct interrupt
  wake/reschedule, not a status-dependent event post in `FUN_0000d828`.

The clean discriminator is a genuine completion capture: one-shot read
`0x27200900`/`0x27200904` at the completion IRQ (kernel-side, not host polling),
plus an execution trace of the firmware callback with the dual framing fixed.
That would settle both the IP-to-line routing and whether firmware or hardware
owns the W1C ack.

## 4. `FUN_00008c68` and the framing collision

Natural publish execution repeatedly enters the separate overlay helper at VMA
`0x8c98`, file `0x8d98`, and returns at VMA `0x8d50`, file `0x8e50`
(`exec_trace_goalive_50000_52400.log:205-280,373-448,540-615,708-783,874-949`).
So iter25 was correct to give that helper `+0x100` bytes.

The post-wake service routine, however, enters at VMA `0x8c6c` in the BASE
stream. Its zero-bit branch at VMA `0x8c8b` targets VMA `0x8cae`. The coherent
BASE tail is:

| VMA | File (`+0x5c`) | Instruction |
|---:|---:|---|
| `0x8ca5` | `0x8d01` | `L8UI a9,[a8]` |
| `0x8ca8` | `0x8d04` | `AND a9,a9,a6` |
| `0x8cab` | `0x8d07` | `S8I a9,[a8]` |
| `0x8cae` | `0x8d0a` | `ADDI a8,a8,0x60` |
| `0x8cb1` | `0x8d0d` | `ADDMI a4,a4,0x1000` |
| `0x8cb4` | `0x8d10` | `ADDMI a5,a5,0x1000` |
| `0x8cb7` | `0x8d13` | `WSR PS,a3` |
| `0x8cba` | `0x8d16` | `RETW.N` |

Source: `m2c_probe_decode_c68_framings.log:25-32`. The current overlay table
claims all of `[0x8c98,0x8d52)` (`src/firmware/mod.rs:188-225`), and `fetch8`
selects overlays solely from virtual address (`src/firmware/mmio.rs:200-226`).
Consequently the BASE branch to `0x8cae` fetches the unrelated overlay byte
stream, decodes `S8I` at `0x8cae`, then Unknown at `0x8cb1`
(`re_probe_goalive_completion_values.log:24-38`).

Verdict: `FUN_00008c68`'s tail is definitely BASE (`+0x5c`), and the current
iter25 overlay over-claims it. The genuine line-0 scheduler path does enter the
service routine: the earlier BASE-framed observation records
`0x7fe1 -> 0x8c6c`, then returns through the dispatcher and `RFE`
(`isr_observe.log:8-18`). The `0x8cb1` wall is therefore a model framing
collision, not evidence that zero completion state selected an invalid firmware
path.

## Confidence boundary

- **Proved by execution:** helper semantics; no descriptor at `0x27200904..20`;
  `0x14800` is constructed later; `0x0405078c` does not control dispatch;
  line 0 wakes the wait and enters the handler; both conflicting framings are
  genuinely executed.
- **Strongly inferred from open source:** `0x27200800` is a ZDMA/ADMA channel
  base because all channel register offsets align exactly.
- **Unresolved:** Phoenix's authoritative name for the block, its interrupt
  routing to Xtensa line 0, who acknowledges `CH_ISR`, and the hardware side
  effect that must precede the go-alive interrupt.
