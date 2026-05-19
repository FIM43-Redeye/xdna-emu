# Interrupt Subsystem Close-Out (Tier A) — Design

**Date:** 2026-05-19
**Status:** Approved (brainstorming complete; ready for implementation plan)
**Coverage subsystem:** `interrupt` (currently `PARTIAL`, target `Full`)

## Goal

Take the AIE2 (NPU1 / Phoenix) `interrupt` coverage subsystem from `PARTIAL`
to `Full` by building a faithful internal interrupt path — event/error →
L1 → broadcast network → L2 → NPI interrupt line — with full register-surface
fidelity, terminating at the on-NPU firmware boundary. Host-visible delivery
(firmware async-event mailbox) is explicitly out of Tier A scope and tracked
as a separate future subsystem (Tier B).

## Background: why the one-line gap string is misleading

The coverage artifact records:

> `interrupt: PARTIAL -- missing: L2 23-reg surface exhaustiveness + privilege gating`

Investigation of the authoritative sources (aie-rt `interrupt/`, AM025
register database, xdna-driver) showed this is misleading in both directions:

- **The "23 registers" are already modeled.** The shim interrupt surface in
  AM025 is exactly 23 registers: 18 L1 (`Interrupt_controller_1st_level_*`,
  9 per switch × A/B, offsets `0x35000`–`0x35050`) + 5 L2
  (`Interrupt_controller_2nd_level_*`, `0x15000`–`0x15010`). `l1.rs`/`l2.rs`
  already model every one of those offsets with correct write-1-to-clear,
  enable/disable→mask, and read-only-mask semantics. Register *offset*
  exhaustiveness was never the gap.

- **The real gap is integration, which the string omits.** Three concrete
  holes:
  1. **Read path not routed.** `effects.rs` routes register *writes* to the
     controllers (`:353-360`), but there is no read routing and no caller of
     `read_register`. The controllers are effectively write-only models —
     write-1-to-clear and enable/disable→mask reflection are invisible to a
     guest read.
  2. **No stimulus driver.** `signal_event` (L1) and `signal_interrupt` (L2)
     have zero non-test callers. Nothing in the event/error pipeline fires an
     event into L1; nothing propagates an L1 output to L2.
  3. **L1→L2 broadcast hop unmodeled.** The functional core of interrupt
     delivery does not exist.

- **Privilege gating is one register.** The only privileged L2 register is
  `Interrupt_controller_2nd_level_interrupt` (NoC routing; aie-rt
  `_XAie_PrivilegeSetL2IrqId`). The project already has a documented policy
  ("NPI privileged register access is driver-side privilege; emulator gives
  unrestricted access: accepted out of scope" — the `noc` domain, CLAUDE.md).
  `shim_mux` and `control_packets` both reached `Full` with explicitly
  scoped-out sub-features plus rationale, so precedent supports formally
  scoping privilege rather than enforcing it.

## Host-boundary finding (scope decomposition)

xdna-driver research established that on Phoenix/NPU1 the AIE L1/L2 shim
interrupt **never reaches the x86 host directly**. It terminates at NPI
interrupt lines consumed by on-NPU MGMT/ERT firmware, which then synthesizes a
mailbox async-event message; the host only ever sees the mailbox MSI-X
(`mailbox_irq_handler`; AIE errors arrive via `aie2_error_async_cb` as async
mailbox messages). The emulator has no firmware/mailbox model — it shortcuts
the MGMT/ERT layer with a synchronous completion model (`xdna_emu_run()` runs
until cores halt / syncs satisfy, returns status through the FFI; the
`XDNA_EMU_MAILBOX_LATENCY` knob is a cycle-accounting fudge, not a mailbox).

This decomposes "interrupt to Full" into three tiers:

| Tier | What | Scope |
|------|------|-------|
| **A — AIE interrupt path** | event/error → L1 → broadcast → L2 → NPI line asserted; reads routed; 23-reg semantics exact; privilege scoped-out. Terminates at the firmware-notify boundary. | **This spec.** Fully derivable from aie-rt + AM025; no HW budget. Closes the `interrupt` coverage gap. |
| **B — Host error delivery** | Synthesize the driver-facing async-event mailbox message so `aie2_error_async_cb` sees AIE errors. | **Tracked, not built.** Requires a firmware/mailbox model the emulator deliberately lacks — a separate subsystem with its own spec/plan cycle. |
| **C — Command completion** | "Run done" visible to XRT. | **Already works** via the synchronous model; no AIE interrupt involved. Not part of this subsystem. |

Tier A is the chosen scope. Tier B is recorded as a tracked follow-up,
mirroring how `control_packets` reached `Full` with the `SLVERR_Block` bit
tracked to the `noc` domain.

## Architecture (Approach 3: broadcast transport, explicit interrupt sink)

A faithful internal interrupt path using the existing broadcast network as
transport plus a new, named interrupt sink. End to end:

```
event fires (Event_Generate OR hardware error event)
        |  [tap, shim tiles]
        v
L1.signal_event(switch, event_id)        <- offered to BOTH switches A & B;
        |  each filters by its own IRQ_EVENT slot config + enable mask
        v  (latched -> L1 drives its IRQ_NO broadcast line)
push IRQ_NO onto tile.pending_broadcasts <- L1 is a 2nd independent producer
        |  into the SAME pending_broadcasts / propagate_broadcasts transport
        v
propagate_broadcasts() BFS               <- existing; honors
        |  per-tile delivery loop            EVENT_BROADCAST_BLOCK masks
        v
[NEW SINK] if dest is ShimNoc & has l2_irq:
        L2.signal_interrupt(channel)     <- channel == broadcast id == IRQ_NO
        |  L2 enable-mask gating decides latch
        v
L2 status latched -> NPI interrupt line asserted   <- TERMINAL (fw boundary)
        |
        +--> Tier B (firmware async-event -> driver) -- TRACKED, not built
```

Channel identity is consistent end to end — L1 `IRQ_NO` (4-bit broadcast id
0–15) == broadcast-network channel == L2 input channel — so there is no
mapping table; this is asserted as an invariant with a test.

Approach 3 was chosen over (1) "ride the broadcast network, tap inside
trace-notify" and (2) "direct column-local L1→L2 call". Approach 1 is equally
faithful but smears interrupt logic into trace-notify; Approach 2 is simplest
but bypasses real broadcast-network semantics (block masks, multi-tile
fan-in), violating the project's "match real hardware, no simplified
approximations" rule. Approach 3 keeps the real broadcast transport (including
block-mask gating) while isolating the interrupt sink as a named, independently
testable seam that mirrors the existing "broadcast arrival notifies trace
units" pattern.

## Components

| File | Change | Responsibility |
|------|--------|----------------|
| `src/device/interrupts/l1.rs` | minor | Controller already models all 18 L1 regs + `signal_event`. Verify switch-independent latch semantics; no structural change expected. |
| `src/device/interrupts/l2.rs` | minor | Already models 5 L2 regs + `signal_interrupt`. No structural change expected. |
| `src/device/state/effects.rs` | **primary** | (a) L1/L2 **read-path routing** mirroring write routing at `:353-360`; (b) **event→L1 tap** at the event-fire site + the error-event entry; (c) **broadcast→L2 sink** in `propagate_broadcasts` per-tile delivery (`:444+`), parallel to the existing trace-unit notify. |
| `src/device/interrupts/mod.rs` | small | Any new helper/constant (e.g. switch-offer); document the wired path in the module doc. |
| `src/device/events/` (EventModule) | small | Ensure hardware error events enter the `EventModule` (the tap point) — see Risk below. |
| `crates/xdna-archspec/src/coverage/units.rs` | small | `interrupt` `Partial{…}` → `Modeled { completeness: Full }`; narrative rewrite citing Tier B as tracked follow-up. |
| `docs/coverage/aie2/*` | regen | Regenerated artifact; only the interrupt line changes; `implementation-gaps.md` drops the interrupt PARTIAL row. |
| `docs/superpowers/findings/2026-05-19-interrupt-tier-b-firmware-delivery.md` | new | Records the xdna-driver host-boundary contract for the future Tier B subsystem. |

The bulk of the work is `effects.rs` wiring; the controllers themselves are
essentially done. The interrupt sink is a named seam in
`propagate_broadcasts`, never interrupt logic smeared into trace-notify.

## Data flow (concrete)

The canonical event sink is the per-tile **`EventModule`**
(`src/device/events/mod.rs`): `generate_event(event_id)` is the unified entry,
`drain_pending()` the observable; error-group semantics already exist
(`group.rs`: `EVENT_GROUP_ERRORS_*` hw_ids + masks).

1. **Event fires** → `EventModule.generate_event(id)` (Event_Generate
   register write already routes here; errors — see Risk).
2. **Tap (shim tiles only):** after the event-fire hook observes the event,
   offer `id` to `l1_irq.signal_event(SwitchId::A, id)` **and**
   `signal_event(SwitchId::B, id)`. Each switch independently slot-matches
   against its own `IRQ_EVENT` config and gates on its own enable mask —
   faithful to the two-independent-switches hardware.
3. **L1 latch → broadcast producer:** if either switch returns
   `Some(interrupt_id)`, push that switch's `IRQ_NO` (4-bit broadcast id)
   onto `tile.pending_broadcasts` — L1 as a second independent producer into
   the existing transport, alongside the EventModule's broadcast-channel
   mapping.
4. **Transport:** existing `propagate_broadcasts` BFS carries it, honoring
   `EVENT_BROADCAST_BLOCK_*` masks.
5. **Sink (new, named seam):** in the per-tile delivery loop, if the
   destination tile is `ShimNoc` with `l2_irq`, call
   `l2.signal_interrupt(channel)`. L2 enable-mask gating decides the latch.
6. **Terminal:** L2 status latched → `pending_host_interrupt()` true → NPI
   line asserted, observable via the now-read-routed L2 `Interrupt`/`Status`
   registers. Tier B forward-pointer documented here.

## Edge cases & error handling

- **Highest risk — error-event entry.** Hardware error events today reach
  trace units directly via `core_trace.notify_event(INSTR_ERROR, …)` (e.g.
  `interpreter/core/interpreter.rs:21`), *not* uniformly through
  `EventModule.generate_event`. If left as-is, only `Event_Generate`-driven
  events would feed L1 and the error path — the primary real consumer per the
  driver research — would silently not work. **Resolution:** the design
  requires hardware error events to also enter the `EventModule` (the tap
  point). The implementation plan carries an explicit
  investigate-then-wire task for this; it must not be hand-waved. The error
  integration test (matrix item 8) is the proof: if it cannot be made to pass,
  the error wiring is incomplete by definition.
- **Switch A/B double-latch.** An event matching slots in both switches
  latches both independently and may drive two broadcast ids — correct per
  hardware (two independent L1 switches); not deduplicated.
- **Channel/id identity.** L1 `IRQ_NO` == broadcast channel == L2 input
  channel (all 0–15). No mapping table; asserted as an invariant test.
- **Block masks.** Broadcast-blocked directions prune L1→L2 like any
  broadcast — already handled by `propagate_broadcasts`; a test verifies a
  blocked path prevents the L2 latch.
- **Disabled-channel drop.** L2 with the channel masked off must not latch
  (existing `signal_interrupt` mask gate) — explicit test.
- **Privilege.** `Interrupt_controller_2nd_level_interrupt` writes accepted
  unrestricted; a doc-comment records the deliberate scope-out (driver-side
  privilege; emulator unrestricted) citing the noc/shim_mux precedent. No
  enforcement code.
- **Non-shim tiles.** L1/L2 exist only on shim tiles; tap and sink are
  guarded by `is_shim()`/`ShimNoc`, exactly as the existing write routing is.

## Testing strategy

Bridge tests do not exercise interrupts (no existing kernel configures L1/L2
or raises interrupts), so validation is **Rust integration tests at the
`DeviceState`/`effects` level** — drive register writes to configure the
controllers, fire an event, assert the full chain. `cargo test --lib` is the
gate. Matrix:

1. **Read-path routing** — every one of the 23 register offsets observable;
   write-enable→read-mask reflects; status write-1-to-clear; read-only mask
   write ignored.
2. **Event→L1** — configure `IRQ_EVENT` slot + enable on switch A,
   `Event_Generate` the mapped event → L1 status latches, `IRQ_NO` broadcast
   queued.
3. **Switch independence** — A-only configured → only A latches; both → both.
4. **Broadcast transport** — L1 latch → `propagate_broadcasts` → shim-NoC L2
   latches on the matching channel.
5. **Block-mask fidelity** — program `EVENT_BROADCAST_BLOCK` on the L1→L2
   direction → L2 does not latch.
6. **L2 mask gating** — channel masked off → no latch; enabled → latch; NPI
   line asserted, observed via read-routed L2 `Interrupt`/`Status`.
7. **Channel-identity invariant** — `IRQ_NO` == broadcast channel == L2
   input, as a probe test.
8. **Error path (high-risk item)** — synthetic hardware error event enters
   `EventModule` → full chain to L2. Proves the error-wiring resolution.
9. **Privilege scope** — write to `Interrupt_controller_2nd_level_interrupt`
   accepted unrestricted (documents the scoped behavior).

**"Full" success criteria:** all 23 registers read/write-observable with
correct semantics; the complete event *and* error → L1 → broadcast → L2 → NPI
chain has passing integration tests including block-mask fidelity; coverage
marker flipped; Tier B tracked.

## Coverage marker & Tier-B tracking

- `units.rs` `interrupt`: `Modeled { completeness: Partial { … } }` →
  `Modeled { completeness: Full }`. Narrative rewritten — L1 (18 reg, 2
  switches) + L2 (5 reg) fully modeled, read+write routed, event/error → L1 →
  broadcast → L2 → NPI wired and tested, block-mask honored, privilege
  scoped-out (driver-side) per the noc/shim_mux precedent. **Tier B named as
  the tracked cross-domain follow-up**, the control_packets→SLVERR_Block→noc
  pattern.
- Regenerate `docs/coverage/aie2/*`; only the interrupt line changes;
  `implementation-gaps.md` drops the interrupt PARTIAL row.
- **Tier B durably captured** in
  `docs/superpowers/findings/2026-05-19-interrupt-tier-b-firmware-delivery.md`
  recording the xdna-driver host-boundary contract (mailbox I2X ring, IOHUB
  interrupt register at `i2x_head_ptr_reg + 4`, `aie2_error_async_cb`
  async-event path, TDR relation), so the future firmware/mailbox subsystem
  starts from recorded derivation rather than re-investigation. The interrupt
  narrative forward-points to it, making Tier B as discoverable as the §8
  OPEN items.

## Sources / derivation

- **aie-rt** `driver/src/interrupt/` (branch `xlnx_rel_v2025.2`): L1/L2 API
  surface, register sequences, L1↔L2 relationship, `_XAie_PrivilegeSetL2IrqId`
  privilege wrapper.
- **AM025 register database** (`mlir-aie/.../aie_registers_aie2.json`): the
  23-register shim interrupt surface, offsets, bit fields.
- **xdna-driver** (`/home/triple/npu-work/xdna-driver/`): host-boundary
  contract — establishes that AIE interrupts are firmware-mediated, scoping
  Tier B out of Tier A.
- All emulator code is original; the above are read-only behavioral
  references per the project source-derivation policy.

## Out of scope (explicit)

- **Tier B** — firmware/mailbox async-event delivery to the host driver.
  Tracked; separate subsystem.
- **Tier C** — command-completion signaling. Already handled by the
  synchronous completion model.
- **Privilege enforcement** — deliberately scoped out (driver-side concern),
  documented per precedent.
- **AIE1 / AIE2P interrupt variants** — NPU1 (AIE2) only, consistent with the
  rest of the emulator's primary target.
