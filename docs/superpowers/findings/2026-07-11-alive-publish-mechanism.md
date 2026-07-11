# FW_ALIVE publish mechanism: neither proposed CPU-store path is proved

Date: 2026-07-11  
Target: Phoenix/NPU1 firmware `1502_00/npu.dev.sbin`  
SHA-256: `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`

## Verdict: other

Neither (i) nor (ii) survives the current frontier trace as stated.

- The `0x5044` function does store little-endian `0x030bb000` through virtual
  addresses `0..3`, but the live DTLB translates those addresses to physical
  `0..3`, not `0x030bf000`. The same local word is later loaded at `0x897d`
  and `0x89b1` and consumed as the channel pointer. This is a live local/global
  pointer slot in the observed address space, not an executed direct store to
  the driver's `FW_ALIVE_OFF` address.
- The current frontier-extended natural boot delivers no line-0 interrupt. It
  reaches the old `0x7fe1 -> 0x8c6c` service landmarks with `EXCCAUSE=1`
  (syscall), not `EXCCAUSE=4` (level-1 interrupt), and performs no store to
  `0x030bf000`.
- The alleged `0x27200800` ZDMA channel is not established. Firmware never
  references that base; it treats `0x27200904+` as packed two-bit fields, which
  is incompatible with the generic ZDMA register semantics assigned to those
  offsets. No open-source Phoenix source maps a generic ZDMA completion to
  Xtensa line 0.

Thus this run does **not** identify a firmware CPU instruction that writes
device address `0x030bf000`. On silicon the slot nevertheless receives
`0x030bb000`; the missing mechanism is outside the executed/modelled path at
the `0x8cb1` frontier. The remaining honest alternatives are a non-DTLB
physical alias/side effect for local PA 0, a PSP/load-time patch absent from the
flat signed image, or later firmware execution beyond the current mapping
frontier. This evidence does not choose among them.

## A. Every current-frontier SRAM-band store

The probe decodes every store through the current natural boot, resolves its
effective address through the same DTLB translator used by execution, and
records it when either EA or PA is in `0x030b0000..0x030c0000`.

There is exactly one:

```text
n=52551 pc=0x8964 STORE4
EA=0x030b27c0 -> PA=0x030b27c0 value=0x00000000
DTLB hit=way 6 entry 0 ring 0
```

There are zero stores whose EA or PA is `0x030bf000`. Execution stops at the
known next mapping wall:

```text
n=53660 Unknown pc=0x8cb1 word=0x61a800
INTENABLE=1 INTERRUPT=0
```

This is a whole-executed-boot result through the current frontier, not a
literal-only search. It includes the already-proved test-local queue-tail
extension `[0xccb4,0xccc1)` and no interrupt or firmware-memory injection.

## B. Candidate writer and its actual gate

The only executed instruction sequence that composes the hardware-observed
value `0x030bb000` is:

```text
n=51765 pc=0x560d Call8 0x5044
n=51766 pc=0x5044 Entry
n=52115 pc=0x50ba L32r a2,[0x31bc] -> a2=0
n=52119 pc=0x50c6 S8i [a2+0] <- 0x00  EA=0 -> PA=0
n=52120 pc=0x50c9 S8i [a2+3] <- 0x03  EA=3 -> PA=3
n=52121 pc=0x50cc S8i [a2+2] <- 0x0b  EA=2 -> PA=2
n=52122 pc=0x50cf S8i [a2+1] <- 0xb0  EA=1 -> PA=1
```

`0x560d -> 0x5044` is an unconditional direct call. There is no interrupt,
status branch, or memory flag between that call and the four stores. Therefore
the `0x5044` local-pointer write itself is not gated on line 0.

The destination is also not statically recoverable as `FW_ALIVE_OFF` from this
image:

- raw `0x31bc` is `0` in both BASE (`file=vaddr+0x5c`) and AT
  (`file=vaddr+0x100`) views;
- the raw image contains no literal `0x030bf000` and no literal `0x030bb000`;
- the static L32R-literal scan over values `0x030b0000..0x030c0000` finds zero
  references.

Literal absence alone cannot exclude a computed address in never-executed
code. The stronger statement supported here is: no direct writer is statically
identified, and no CPU store to `0x030bf000` executes before the current
frontier.

The alleged scratch readback is real on the later calls, with one important
qualification: `0x897d` is called more than once and its first EA is not zero.
The exact consumed values are:

```text
n=50863 pc=0x897d EA=0x00011784 consumed=0x00000000
n=52561 pc=0x897d EA=0x00000000 consumed=0x030bb000
n=52577 pc=0x89b1 EA=0x00000000 consumed=0x030bb000
```

This overturns the stronger Finding-C statement that the four byte stores are
already proved to be the host-visible doorbell. They are proved to create a
live pointer slot at local PA 0. A dual-visible hardware alias remains possible,
but it is not proved by the firmware MMU or the on-disk image.

## C. DTLB translation of VA 0 at `0x50c6`

At the first byte store:

```text
DTLBCFG=0x00030000
PTEVADDR=0x3c000000
RASID=0x04030201
VA0 lookup = TlbHit { wi: 6, ei: 0, ring: 0 }
DTLB[6][0] = { vaddr: 0, paddr: 0, asid: 1, attr: 3 }
VA 0 -> PA 0
```

The executed startup TLB operations agree. The relevant early write is
`n=995 pc=0x2e6 WDTLB at=0x00000007 as=0x20000005`, which maps virtual
`0x20000000` to physical 0. Seven `IDTLB` operations then invalidate the
`0x20000000..0xe0000000` region entries. No executed WDTLB installs a paged
VA-0 mapping to `0x030bf000`; the low identity region entry remains the hit at
publication.

This rules out a firmware-programmed DTLB mapping from VA 0 to
`FW_ALIVE_OFF`. It does not rule out an undocumented physical-bus alias after
the Xtensa MMU or a PSP mutation not represented by the file; either would be
external to the translation state observed here.

## D. Natural line-0 reachability and source identity

The current frontier-extended boot raises these exceptions:

```text
EXCCAUSE 1 (syscall): 5
internal window exception 0x1000: 1
EXCCAUSE 4 (level-1 interrupt): 0
```

The landmarks that previously looked like a line-0 service path are reached
naturally, but their consumed cause proves they are the syscall arm:

```text
n=53575 pc=0x8784 EXCCAUSE=1 INTERRUPT=0
n=53639 pc=0x7fe1 EXCCAUSE=1 INTERRUPT=0
n=53640 pc=0x8c6c EXCCAUSE=1 INTERRUPT=0
n=53659 pc=0x8cb1 EXCCAUSE=1 INTERRUPT=0
```

So the old statement “the line-0 ISR path publishes” is **overturned**. The
old `0x7fc4 -> 0x8c6c` trace did not prove an interrupt-gated publisher; on the
current natural boot that shared service path is entered by a syscall.

Go-alive does configure two controller sources before the local-pointer store:

```text
n=51780 pc=0x86f8 a10=0x2e
n=51890 pc=0x86f8 a10=0x2d
source 0x2d -> slot 13 -> record 0x111b4 -> handler 0x5948(arg 0x0d)
source 0x2e -> slot 14 -> record 0x111c8 -> handler 0x5948(arg 0x0e)
```

Xtensa bit 0 is only the aggregate level-1 input. A real interrupt dispatch
would read the controller source ID from `0x272003c4`, map it through
`0x11700+source`, and invoke the selected record. The open-source toolchain and
driver do not name the physical devices behind sources `0x2d` and `0x2e`.

The `0x27200800 ZDMA completion` label cannot be used as that missing name:
firmware never references `0x27200800`; it references `0x27200904` and performs
packed two-bit RMWs across the block. The generic Xilinx ZDMA map would instead
assign `CH_ISR/IMR/IEN/IDS/...` to `+0x100/+0x104/+0x108/+0x10c/...`, with
DMA-done bit `0x400` and platform-specific interrupt routing. No Phoenix source
connects those generic semantics to line 0.

Accordingly there is no exact hardware interrupt stimulus that can honestly be
named from this evidence. “Assert line 0” is under-specified, and “deliver the
ZDMA completion” is unsupported. Sources `0x2d` and `0x2e` are the exact
firmware-configured controller IDs, but their device origins remain unresolved;
neither is proved to gate or perform the `FW_ALIVE_OFF` write.

## Reproduction

The probe is `m2c_probe_alive_publish_mechanism` in
`src/firmware/boot_tests/coherence_mapper.rs`:

```text
XDNA_FW_PROBE=1 XDNA_FW_MAX=100000 cargo test --lib \
  m2c_probe_alive_publish_mechanism -- --nocapture
```

The prior queue discriminator remains reproducible independently:

```text
XDNA_FW_PROBE=1 XDNA_FW_MAX=100000 cargo test --lib \
  m2c_probe_goalive_loop_discriminator -- --nocapture
```

No production `load_m2c`, scheduler, MMU, interrupt model, or firmware memory
was changed.
