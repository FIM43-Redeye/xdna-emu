# Faithful Firmware Task-Completion Model -- Design

> Status: design approved (Maya, 2026-07-06). Next: writing-plans.
> Branch: `feat/m2c-mapping-boot-to-idle`. Issue: #140 (firmware-emulation dream, iter18).

## Background

The in-tree Xtensa firmware interpreter (`src/firmware/`) boots the real NPU
management firmware one wall at a time. The current wall: after the mailbox
"post", the firmware enters its cooperative RTOS `task_dispatcher` (`0xd7f0`) and
recurses forever because the running task's **done-flag** `[task+0x30]`
(=`0x9070` at boot) is never set. `boot_to_idle` (`src/firmware/mod.rs:188`) never
reaches idle.

A full session of falsifiable experiments (iter18, findings
`docs/superpowers/findings/2026-07-06-iter18-phase0-interrupt-wiring.md`)
resolved *what* sets that done-flag:

- **Not the firmware itself.** The store-search found no firmware code that ever
  writes offset `+0x30`. An external agent must write it.
- **Not a register handshake.** `m2c_probe_force_ack` performed every
  xdna-driver-derived host ack (i2x head advance, intr clear, tail zero/advance,
  doorbell) and all were byte-for-byte inert: the scheduler recursion polls only
  local memory (poll-map), never the mailbox registers.
- **Not a delivered interrupt.** The firmware arms exactly one interrupt
  (level-1 bit 0) then locks `INTLEVEL` at 2 inside the recursion's critical
  section and never yields to the idle `waiti`, so a doorbell is undeliverable.
- **It is an async write to firmware-LOCAL memory by a hardware agent** (shape
  ii). `m2c_probe_force_done` proved that writing `[task+0x30]=1` directly
  unwinds the recursion into a real RTOS context switch. force-done is the
  faithful *stub*; the only unfaithful thing about it is that it fires at the
  dispatcher's check PC rather than being triggered by the real event.

**This design replaces the force-done stub with the faithful mechanism:** the
completion is triggered by the firmware's mailbox **post** (its store to the i2x
tail register), and delivered as a direct local-memory write of the current
task's done-flag.

## Goal

`boot_to_idle` advances past the `task_dispatcher` recursion along the **real**
post-completion path (the task selected from real scheduler state, not
force-done's artificial pick), driven by a faithful model of the host mailbox
consumer and the NPU's local completion agent. Then walk that real path and
clear the next genuine wall.

## Design principles (binding)

1. **No latency calibration.** The completion has zero modeled latency and no
   tuning knob. The store-watch proved the firmware never re-clears `0x9070`, so
   a completion written any time after the post simply sticks -- there is no
   arm/clear race that would force a delay. We are not hiding a hardware timing;
   a real latency would only ever *emerge* from modeling the transfer itself.
   Any future temptation to inject a hardware-measured delay is a stop-and-
   reconsider point, per Maya's "focus on real emulation" directive.
2. **Derive from the toolchain.** The mailbox ring layout, register offsets,
   header format, and ack sequence are derived from xdna-driver
   (`amdxdna_mailbox.c`, `aie2_pci.c`, `aie2_msg_priv.h`), not guessed. Cited
   inline as the source of the behavioral fact.
3. **Two agents, matching two hardware blocks.** The host consumer and the local
   completion agent are separate types even though the completion agent's only
   current action is one write -- the split is what lets each grow independently
   later (host protocol vs. DMA/latency modeling).
4. **Layer 1 to 100%, projections marked.** Build the full i2x-consume protocol
   the boot path exercises. Mark every place a later layer will plug in with a
   `// PROJECTED Layer N:` comment so the growth points are visible, not
   rediscovered.

## Architecture: two agents

Real silicon has two distinct blocks between the firmware's post and the
done-flag write; the model mirrors them as two types in a new focused module
`src/firmware/host_mailbox.rs`:

### Agent 1 -- Host mailbox consumer (`HostMailboxConsumer`)

Models the x86 host servicing a fw->host (i2x) message, per xdna-driver's
`mailbox_rx_worker` / `mailbox_irq_acknowledge`:

1. Detect the post: an **advance** of the i2x tail register `0x27200170` (edge on
   the tracked value; not the magic `0xf18` -- any forward move).
2. Read and validate the 16-byte message header from the i2x **ring buffer** at
   the head offset (magic top-byte `0x1D`, `MSG_PROTOCOL_VERSION=0x1`, opcode,
   total_size), per `amdxdna_mailbox.c:121-130`. Modeling the ring buffer is in
   scope for this effort (see "Ring buffer memory model").
3. Acknowledge: write i2x **head** `0x27200174` = tail (ring consumed), intr
   `0x27200178` = 0. Faithful to the driver; **inert to the stuck boot** (the
   firmware never reads these back -- force-ack proved it), but the real protocol
   that post-idle paths will use.
4. Hand the consumed request to the completion agent.

### Agent 2 -- Local completion agent (`CompletionAgent`)

Models the NPU's local mailbox-completion hardware that writes the request's
done-flag into firmware-local SRAM (shape ii). On a consumed request:

1. Read the current-task pointer live from the scheduler global:
   `task = load_local32(0x2250 + 0x28)` (= `0x9040` at boot; the exact address
   the dispatcher itself uses at `0xd81a`).
2. Compute the done-flag address `done = task + 0x30` (= `0x9070`).
3. Write `store_local32(done, 1)` -- zero latency, once per consumed request.

Keeping the write in its own agent is deliberate: latency/DMA modeling, if ever
needed, grows here without touching the ring protocol.

### Causal chain (all synchronous, zero-latency)

```
fw stores i2x tail 0x27200170 (the POST)
    -> HostMailboxConsumer detects the tail advance
       -> reads+validates the i2x header from the ring buffer
       -> acks: head 0x174 = tail, intr 0x178 = 0
       -> hands the request to CompletionAgent
          -> task = load_local32(0x2278); done = task + 0x30
          -> store_local32(done, 1)
    ... next dispatcher check at 0xd828 sees [task+0x30] != 0 -> unwinds
```

## Integration point

The `boot_to_idle` step loop (`src/firmware/mod.rs:188`) owns a `HostMailbox`
(holding both agents) and `tick()`s it once per instruction, after `cpu.step`.
`tick` reads the i2x tail (side-effect-free), compares to the last-seen value,
and on an advance runs the consume + completion chain. The bus stays generic;
`mod.rs` gains only the `tick` call and the field.

The `HostMailbox` is created disabled by default and enabled explicitly for the
boot-to-idle path, so unrelated firmware unit tests that step the CPU are
unaffected. (Concretely: `boot_to_idle` constructs and ticks it; other callers
of `cpu.step` do not.)

## Ring buffer memory model

Reading the message header requires the i2x ring buffer to be backed memory so
the firmware's own posted bytes are readable. The mailbox aperture
(`0x27000000..0x28000000`) is already backed (`Bus.mailbox`, grows lazily); if
the i2x ring buffer's base (`i2x_buf`, from the `mgmt_mbox_chann_info` struct the
firmware writes to SRAM) falls inside it, the header is already readable and no
new region is needed. If it falls in the currently-unmodeled `0x08a00000..
0x08b00000` gap (the observed payload pointer `0x08a00ff0` routes to `System` and
drops writes today), the plan extends the bus with a backed region for it.

**Open item pinned for implementation (RE, not a design gap):** the exact i2x
ring buffer base and the head/tail-to-buffer-offset arithmetic. Derived from
`amdxdna_mailbox.c` (ring layout) + a boot probe that reads back the
`mgmt_mbox_chann_info` struct the firmware writes. The plan's first task is this
RE + a memory-region decision; everything downstream consumes its result.

## Derived constants (with sources)

| Constant | Value | Source |
|----------|-------|--------|
| i2x tail reg (fw writes; the POST) | `0x27200170` | observed + `aie2_pci.c:376` reg map |
| i2x head reg (host ack) | `0x27200174` | xdna-driver `mailbox_set_headptr` |
| i2x intr reg (host writes 0) | `0x27200178` | `aie2_pci.c:376-379` (`head_ptr_reg + 4`) |
| message header | 16 B `{total_size, sz_ver, id(magic 0x1D), opcode}` | `amdxdna_mailbox.c:121-130` |
| protocol version | `0x1` | `MSG_PROTOCOL_VERSION` |
| scheduler global -> current task | `load_local32(0x2250 + 0x28)` = `0x9040` | dispatcher `0xd81a` (findings) |
| task done-flag offset | `+0x30` -> `0x9070` | dispatcher `0xd828` `l32i.n a10,[a4+0x30]` |

## Layer 1 scope and projection markers

**In scope (build to 100%):** i2x post detection, header read+validate, i2x
acknowledge (head+intr), ring buffer memory model, direct done-flag completion,
re-arm per post.

**PROJECTED (mark in code as `// PROJECTED Layer N:`, do not build until a real
path needs it):**

- **x2i host->fw response ring** -- when a post-idle path first *reads* x2i.
- **opcode dispatch** -- when the completion or response must depend on message
  type (`GET_PROTOCOL_VERSION 0x301`, `GET_FIRMWARE_VERSION 0x108`,
  `ASSIGN_MGMT_PASID 0x103`, `SET/GET_RUNTIME_CONFIG 0x10A/0x10B`,
  `REGISTER_ASYNC_EVENT_MSG 0x10C`; `aie2_msg_priv.h`), not just "ack + complete".
- **completion via response-DMA** vs the direct `[task+0x30]` write -- if a
  future path shows the done-flag arriving through a response buffer.
- **completion-address from message contents** -- if header parsing reveals the
  request carries the done-flag address explicitly, prefer it over the
  scheduler-global read (more faithful, handles concurrent posts). Until then the
  scheduler-global read is what the evidence supports.
- **done-flag value as status** -- we write `1` (dispatcher only does `beqz`);
  if a downstream consumer reads it as a status code or pointer, revisit.

## Error handling / guards

- **Scheduler not up:** if `load_local32(0x2278)` is `0` or out of the local
  range, consume the post (protocol ack) but skip the completion write -- no
  valid task to complete. A post before the scheduler initializes must not write
  a garbage done-flag address.
- **Header validation failure:** if the magic top-byte is not `0x1D` or the
  protocol version is not `0x1`, log and consume without completing (a
  malformed/unexpected message is not a boot-completion request). This surfaces
  a wrong ring-buffer base early rather than silently completing on noise.
- **Idempotent re-arm:** completion fires once per tail advance. A tail that does
  not move drives nothing.

## Testing

**Unit (no firmware image; synthetic `Bus`):**
- A tail advance at `0x27200170` with a valid header and a valid scheduler global
  drives exactly one `store_local32(task+0x30, 1)`.
- Second distinct post re-arms and completes again.
- Zero / out-of-range scheduler pointer: post consumed, no completion write.
- Invalid header magic: consumed, no completion write.
- i2x head/intr acknowledged (head == tail, intr == 0) after a consume.

**Boot integration (`XDNA_FW_PROBE`-gated, real `.sbin`):**
- `boot_to_idle` with the agent enabled advances past the `0xd7f0` recursion
  along the real path; record and assert where it next stops.

## Next-wall follow-through (per approved scope: mechanism + next wall)

Once the real path is unblocked, characterize where it stops and clear the next
**bounded** wall. Caveat carried from the findings: `0xd900` (S32C1I, already
landed `d9d7401b`) and `0xd903` (xt_format1 FLIX bundle) were reached on
force-done's *artificial* task-switch path. The faithful completion selects the
task from real scheduler state, so the real path may or may not hit that same
atomic helper. Build the mechanism first, observe the real path, then clear the
next genuine wall -- likely xt_format1 (derivable from the same
`xtensa-modules.c` FLIX tables as format2, per
`2026-07-05-m2c-flix-bundle-decode-design.md`), confirmed on the real path
before implementing rather than off the artificial one.

## What this is not

- Not an interrupt/doorbell delivery path (ruled out: undeliverable in the
  spin). The Phase-1 interrupt machinery stays as-is (landed, inert here).
- Not a latency model (principle 1).
- Not the full mailbox subsystem -- x2i, opcode dispatch, and response-DMA are
  projection-marked, not built.
