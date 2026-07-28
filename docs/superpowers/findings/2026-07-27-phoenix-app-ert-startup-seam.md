# Phoenix APP-ERT Startup Seam

**Date:** 2026-07-27

**Status:** Proven diagnostic boundary; no production behavior implemented

## Verdict

The pinned context command is not first blocked at the BAR4-tail wake or at
APP-ERT opcode dispatch. APP-ERT is already blocked during `CREATE_CONTEXT`,
waiting for an external startup service to complete two syscall-112 requests.

The first request, opcode `0x10`, supplies a 104-byte arena containing four
firmware ring descriptors. That arena remains zero in the emulator. The second
request, opcode `0x16`, returns a status word. Event `0x10000` is the completion
signal for both requests.

Management-controller source 37 is the later context-mailbox notification:
it authentically delivers APP-ERT event 4. It cannot dispatch a context command
until the startup service has populated the ring-descriptor arena.

## Pinned Inputs

- Phoenix/NPU1 firmware: `1502_00/npu.dev.sbin`
- Firmware version: `5.5.391`
- Driver-shaped lifecycle: management initialization, `CREATE_CONTEXT`,
  `MAP_HOST_BUFFER`, then `CHAIN_EXEC_NPU` (`0x18`)
- Frozen Chess `add_one_using_dma` instruction stream

## Evidence

### APP-ERT is blocked during CREATE_CONTEXT

The real `CREATE_CONTEXT` execution enters:

```text
0x08b04554  APP-ERT task
0x08b04428  event_wait
```

At the resulting natural scheduler `waiti`, APP-ERT's MERT object at local
`0x21cc` is in waiting state 5 with wait mask `0x10000`. This occurs before the
test publishes any context command.

Static disassembly of the APP-ERT entry confirms the first wait:

```text
0x08b04554  clear 0x68-byte arena at stack + 0xb8
0x08b04569  request target = 0xff06
0x08b0456f  request header = 0x00100010
0x08b04575  request payload pointer = arena
0x08b0457c  syscall-112 wrapper
0x08b0457f  event mask = 0x10000
0x08b04586  event_wait(mask, blocking)
```

The `0x68`-byte arena is four adjacent 24-byte ring descriptors plus eight
bytes of surrounding state. The later event loop uses:

```text
arena + 0x00  event-4 input queue
arena + 0x18  event-5 input queue
arena + 0x30  event-4 output queue
arena + 0x48  event-5 output queue
```

Each 24-byte queue has the shape exercised by the firmware's own queue helpers:
producer cursor pointer, consumer cursor pointer, optional notification
pointer, ring-buffer pointer, capacity, and notification argument.

### The second startup request returns a status

After the first completion, APP-ERT calls `0x08b0596c`:

```text
request target = 0xff06
request header = 0x00100016
request payload[0] = &result
request payload[1] = 0x08a80014
result = 1
syscall-112 wrapper
event_wait(0x10000, blocking)
return result
```

Live translation pinned the result slot at virtual `0x08a00e4c`, physical
`0x00026e4c`. It contains the initialized failure sentinel `1` while the
request is pending.

A diagnostic-only completion that wrote result `0` and delivered event
`0x10000` made APP-ERT survive startup and enter its normal
`event_wait(0xffffffff, 1)` loop. An event without the result write made the
startup call return failure instead. The completion contract therefore
contains both data and an event; the event alone is insufficient.

### Source 37 is the context event, but the input queue is absent

After `CREATE_CONTEXT`, controller source 37 has dynamic metadata:

```text
source       37 (0x25)
selector     5 (APP-ERT object 0x21cc)
event index  4
```

An explicit source-37 assertion follows the authentic firmware path:

```text
controller dispatcher
  -> generic ISR 0x5948
  -> MERT event delivery 0xd034
  -> controller acknowledgement and re-enable
  -> APP-ERT event_wait returns 4
  -> APP-ERT selects its event-4 input queue
```

After the two diagnostic startup completions, source 37 does wake APP-ERT and
the event wait returns exactly 4. APP-ERT then calls its queue-empty helper at
`0x08b0e2e8`. The event-4 input descriptor is still all zero, so the helper
returns empty and APP-ERT goes back to sleep. The context X2I head remains zero;
the posted command is not consumed.

This refutes both of these incomplete models:

- BAR4 tail publication alone is enough to dispatch APP-ERT.
- Source 37 assertion alone is enough to dispatch APP-ERT.

### CREATE_CONTEXT already owns the canonical CQ data

The real firmware-generated `CREATE_CONTEXT` state contains the returned CQ
pair contiguously at local `0x14510..0x1452c`:

```text
0x14510  x2i head  0x030da004
0x14514  x2i tail  0x030da000
0x14518  x2i buf   0x030aa000
0x1451c  x2i size  0x00000400
0x14520  i2x head  0x030db004
0x14524  i2x tail  0x030db000
0x14528  i2x buf   0x030ab000
0x1452c  i2x size  0x00000400
```

The emulator therefore does not need invented queue addresses. What remains
unproved is the external startup service's exact conversion of this CQ record
into APP-ERT's four internal ring descriptors, including notification fields
and the event-5 pair.

## Corrected Boundary

The next missing component is the service reached by syscall 112 for target
`0xff06`, at least for startup operations `0x10` and `0x16`:

```text
APP-ERT startup request
  -> missing service consumes request
  -> operation 0x10 populates four ring descriptors
  -> operation 0x16 writes status
  -> service posts APP-ERT event 0x10000
  -> APP-ERT enters mailbox loop
  -> source 37 / event 4 can expose the posted context request
```

It is not yet established whether that service is another firmware execution
domain, a hardware IPC endpoint, or a combination. It must not be implemented
as a direct context-tail hook, a synthetic APP-ERT response, or a bare event
injection.

## Next Decision

Before production code, identify the owner and externally observable contract
of target `0xff06` operations `0x10` and `0x16`. The smallest acceptable model
will derive its queue addresses from the firmware-produced context record and
will reproduce the real completion data plus event ordering. Event 4 and
source 37 remain downstream of that startup contract.
