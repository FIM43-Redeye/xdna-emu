# Phoenix Post-Alive Observability Boundary

Date: 2026-07-27

Target: Phoenix / NPU1 (`1022:1502`)

Firmware:
`/usr/lib/firmware/amdnpu/1502_00/npu.dev.sbin`,
SHA-256
`d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`

## Verdict

One ordinary management transaction now grounds the host-visible control
envelope when combined with the loaded driver's source ordering:

```text
BAR2 request copied
  -> BAR4 X2I tail issued
  -> firmware advances X2I head
  -> firmware publishes I2X response
  -> MSI-X 14 / host IRQ
  -> host advances I2X head
```

It cannot ground the controller/Xtensa interior:

```text
unknown controller transition
  -> active source 46
  -> Xtensa interrupt bit 0
  -> firmware event (6,4)
```

The internal states are outside every Phoenix PCI BAR and current firmware
debug interface. End-to-end success plus static firmware analysis corroborates
that a path exists, but does not identify the hardware bridge well enough to
implement it without invention.

## Pinned Host Apertures

Live PCI resources on kernel `7.1.5-custom+`:

| BAR | Host physical range | Size | Device base |
|---|---:|---:|---:|
| 0 | `0x90c00000..0x90c7ffff` | 512 KiB | `0x03000000` |
| 2 | `0x8e20800000..0x8e2083ffff` | 256 KiB | `0x03080000` |
| 4 | `0x90c80000..0x90cbffff` | 256 KiB | `0x030c0000` |

The open driver's NPU1 descriptor and mailbox code pin:

| Boundary | Device address |
|---|---:|
| X2I ring | `0x030bc000` |
| I2X ring | `0x030bd000` |
| X2I tail / head | `0x030ec000` / `0x030ec004` |
| I2X tail / head / IOHUB status | `0x030ed000` / `0x030ed004` / `0x030ed008` |

`mailbox_send_msg()` copies the complete packet with `memcpy_toio()` before
publishing X2I tail and emitting `mbox_set_tail`. The receive worker clears
IOHUB status, drains complete responses, publishes I2X head, and rechecks
status before returning. Relevant sources:

- `../xdna-driver/src/driver/amdxdna/npu1_regs.c`
- `../xdna-driver/src/driver/amdxdna/amdxdna_mailbox.c`
- `../xdna-driver/src/driver/amdxdna/amdxdna_trace.h`

## What the Firmware Evidence Actually Pins

Slot-14 setup calls the event and selector helpers separately:

```text
setup_event(slot=14, class=6, event=4)
setup_selectors(slot=14, value=3)
```

The selector helper expands slot 14 to IDs 14, 38, 108, and 109. The packer
places their two-bit value at:

| ID | Register field |
|---:|---:|
| 14 | `0x27200904[29:28]` |
| 38 | `0x2720090c[13:12]` |
| 108 | `0x2720091c[25:24]` |
| 109 | `0x2720091c[27:26]` |

The numerical writes are verified. Open artifacts do not name these fields
“subordinates” or value 3 a “route”; those former labels were inference.

Separately, the statically recovered source-46 handler path:

- is enabled at `0x27200304`, bit 14;
- is read from the active-source register `0x272003c4`;
- dispatches slot 14 / handler `0x5948`;
- is acknowledged at `0x272003b4`, bit 14;
- clears Xtensa interrupt bit 0 with `WSR INTCLEAR, 1`;
- posts firmware event `(6,4)`.

The enable, active-source, and acknowledgement registers are different roles.
In particular, `0x272003b4` is an acknowledgement bank, not evidence of the
originating pending state.

Static evidence entry points:

- `docs/superpowers/findings/2026-07-11-handler-5948-trace.md`
- `docs/superpowers/findings/2026-07-11-goalive-dispatch-target-and-completion.md`
- `src/firmware/boot_tests/static_tools.rs`
- `build/experiments/firmware-re/alive-publish-reconciliation.log`

## Visibility Ceiling

The controller addresses `0x272003xx` and `0x272009xx`, dispatch tables,
firmware event objects, and Xtensa special registers have no BAR0, BAR2, or
BAR4 mapping. Applying the known management-address translation would place
the controller near device address `0x032003xx`, beyond the published
`0x03000000..0x03100000` host aperture.

The current interfaces therefore cannot observe:

- the first controller transition caused by X2I-tail publication;
- active-source value `0x2e` at the dispatcher read;
- `INTERRUPT & INTENABLE` bit 0 before clear;
- handler argument 14 and event-object identity at the same instant.

Phoenix firmware trace and coredump opcodes were already rejected by the
pinned firmware; see
`docs/superpowers/findings/2026-05-06-npu1-msg-op-capability-survey.md`.
The old `boot_capture` patch samples BAR0 boot registers at coarse, gapped
intervals and cannot observe a post-alive controller event. It must not be
extended into a tight poll.

## Captured Outer Envelope

The loaded in-tree driver does not expose a userspace
`GET_PROTOCOL_VERSION` (`0x301`) trigger. The existing repository telemetry
probe instead issued one legitimate
`DRM_AMDXDNA_QUERY_TELEMETRY(type=PROFILING)` request, which the driver maps to
`MSG_OP_GET_TELEMETRY` (`0x4`). The probe was compiled unchanged against the
loaded kernel tree's UAPI.

Capture tuple:

- kernel `7.1.5-custom+`, source commit
  `d9543a0221781d2a9bc72258c2d38f0fb7453e90`;
- loaded `amdxdna.ko` SHA-256
  `9b403eb8d34f0a66f385e6918bba1ebf86da5b527393280047588196b2d16297`;
- firmware SHA-256
  `d13ff9fb95c6cea40213fa69e5a3465529f00bb67c0984d62343c6e31808fb9e`;
- probe source SHA-256
  `68ab96d9317bf8fffbe6d1da029b48fef94772827a48ed2004f3b43450565ed0`;
- probe binary SHA-256
  `641b1e52f46956730e9526e08b3d8612c78fd763ccaa0915e919547b3575295f`.

Exactly four samples were captured with no lost samples:

```text
141757.492181150  mbox_set_tail  irq=145 id=0x1d00000e opcode=0x4
141757.492279194  mbox_irq_handle irq=145
141757.492289072  mbox_rx_worker  irq=145
141757.492339056  mbox_set_head  irq=145 id=0x1d00000e opcode=0x4
```

The measured wall-clock intervals were:

| Interval | Time |
|---|---:|
| tail trace -> IRQ trace | 98.044 us |
| IRQ trace -> receive worker | 9.878 us |
| receive worker -> head trace | 49.984 us |
| tail trace -> head trace | 157.906 us |

The matching opcode and message ID prove that the one response retired the one
request. The successful response reported telemetry version `1.0`, type 3,
six context-map entries, and returned the profiling buffer to userspace.

The packed request and response bodies are each 16 bytes, so each ring packet
is 32 bytes including the mailbox header. The tracepoints do not include
pointer values or raw ring bytes. The loaded source establishes that the
request copy precedes `mbox_set_tail` and response parsing precedes
`mbox_set_head`; the capture proves the intervening hardware IRQ and matching
response. It does not prove a no-wrap pointer delta.

No raw BAR read or write, controller access, polling loop, module reload, or
hardware reconfiguration was used. The tracepoint channel suffix `145` is the
Linux IRQ number, not firmware MSI-X index 14. These host wall-clock timings
are not an AIE-cycle timing oracle.

Artifacts:

- `build/experiments/firmware-post-alive/20260727-host-envelope/perf.data`
- `build/experiments/firmware-post-alive/20260727-host-envelope/perf-script.txt`
- `build/experiments/firmware-post-alive/20260727-host-envelope/trigger.log`
- `build/experiments/firmware-post-alive/20260727-host-envelope/SHA256SUMS`

The sibling driver tree's unrelated dirty `boot_capture` work was not touched.

## Evidence Required Before Implementing the Bridge

The narrowest sufficient new primitive is a non-halting management-Xtensa
instruction/data/SR trace around one ordinary request, or an authoritative
controller specification. It must pin:

- the first controller transition caused by X2I-tail publication and its
  address/value;
- PC `0x878d` reading active source `0x2e` from `0x272003c4`;
- Xtensa interrupt bit 0 before clear;
- handler `0x5948` with argument 14;
- event-service call `0x595f -> 0xd034` with event `(6,4)` identity;
- `0x272003b4 <- 0x4000` and `WSR INTCLEAR, 1`.

Until then, neither a direct `cpu.interrupt |= 1` seam nor a guessed selector
model is licensed.
