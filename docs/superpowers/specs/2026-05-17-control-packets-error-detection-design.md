# Control-Packet Handler Error Detection -- Design

**Status:** approved (brainstorming, 2026-05-17)
**Closes:** `control_packets` PARTIAL implementation gap (coverage spine)
**Outcome:** `control_packets` -> `Modeled { completeness: Full }`

## 0. Problem

`Tile_Control_Packet_Handler_Status` (compute `0x3FF30`, memtile `0xB0F30`)
is a 32-bit register with four write-1-to-clear sticky error bits. The
emulator wires exactly one of them. The coverage system flags this as
`Modeled{Partial{missing: "Tlast/SLVERR/ID_Parity packet-handler sticky
bits"}}`.

That annotation is itself inaccurate. Per the AM025 extract
(`docs/xdna/am025-compact/core_module/tile_control.txt`) and aie-rt
`XAIEGBL_CORE_VALUE_TILCTRLPKTHANSTA`, the true bit map is:

| Bit | Name | State today |
|-----|------|-------------|
| 0 | First_Header_Parity_Error | not wired |
| 1 | Second_Header_Parity_Error | wired (the only one) |
| 2 | SLVERR_On_Access | not wired |
| 3 | Tlast_Error | not wired |

There is **no `ID_Parity` bit anywhere in AM025**. The comment at
`src/device/tile/mod.rs` (the `pkt_handler_status` doc block, currently
"`[0] ID_Parity_Error`") is erroneous, and the coverage annotation
inherited that error.

This work wires the two genuinely unwired bits with faithful
deterministic detecting paths -- First_Header_Parity (0) and Tlast_Error
(3) -- tightens bit 1 (Second_Header_Parity) to true-parity-only, and
corrects the misnomer in the comments and coverage string.

**SLVERR_On_Access (bit 2) is deliberately deferred to a dedicated
successor plan** (see Section 3.3 and Section 8). Tracing the live path
showed the premise it was scoped on is false: `ControlPacketProcessor::
process()` is dead code at runtime, and the real register-write sink
`DeviceState::write_tile_register` returns `()` and cannot fail -- there
is no faithful place a slave error arises today. Wiring bit 2 honestly
requires repairing that path (a register-access-error model), which is
its own piece of work. This plan therefore closes `control_packets` to
an *accurate, narrowed* `Modeled{Partial}` (missing == exactly SLVERR),
not `Full`. The successor plan closes it to `Full`.

## 1. Scope

In scope:

- Detecting paths + sticky-bit latching for First_Header_Parity (0) and
  Tlast_Error (3).
- Tightening Second_Header_Parity (1) to fire only on a genuine
  opcode-header odd-parity failure (today it fires on every header parse
  failure, including structural ones).
- Correcting the `ID_Parity_Error` misnomer in `tile/mod.rs`,
  `state/effects.rs`, and the `units.rs` coverage seed -- the comments
  document the true 4-bit AM025 register regardless of which bits we
  wire.
- Lockstep regeneration of the three generated coverage artifacts, with
  `control_packets` narrowed to an accurate SLVERR-only `Partial`.

Out of scope:

- **SLVERR_On_Access (bit 2)** -- deferred to the successor plan
  (Section 3.3 / Section 8). The `PktHandlerError` enum ships with three
  variants now; the successor plan adds the `Slverr` variant when it
  wires the path.
- No change to write-1-to-clear (`effects.rs` already covers all 4
  bits) or the register read path (`registers.rs` already returns
  `& 0xF`). Both were built generic; only detection was missing.
- No new regdb-driven extraction -- bit positions are centralized
  literals with an AM025 citation.

## 2. New unit: `PktHandlerError`

New file `src/device/control_packets/status.rs`:

```rust
/// Tile_Control_Packet_Handler_Status sticky-error conditions with a
/// faithful detecting path. Bit positions per AM025
/// Tile_Control_Packet_Handler_Status (regdb fields
/// First_Header_Parity_Error / Second_Header_Parity_Error /
/// Tlast_Error). This is the single source of truth for the bit map;
/// no other site names these positions.
///
/// SLVERR_On_Access (bit 0x4) is intentionally absent: it has no
/// faithful trigger until the successor plan repairs the runtime
/// register-access path. That plan adds the `Slverr` variant here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PktHandlerError {
    FirstHeaderParity,  // bit 0
    SecondHeaderParity, // bit 1
    Tlast,              // bit 3
}

impl PktHandlerError {
    pub fn bit(self) -> u32 {
        match self {
            PktHandlerError::FirstHeaderParity => 0x1,
            PktHandlerError::SecondHeaderParity => 0x2,
            PktHandlerError::Tlast => 0x8,
        }
    }
}
```

Exhaustive match, no `_` arm (project convention -- a new variant must
force a compile error here, which is exactly what forces the successor
plan to handle `Slverr` when it adds it).

## 3. Detection semantics

Odd parity is computed identically everywhere in the codebase as
`word.count_ones() & 1 == 1` (it appears inline in
`src/device/stream_switch/packet_types.rs` `PacketHeader::decode:165`
and `encode:140`). There is no standalone helper today. DRY this: add
`pub fn odd_parity_ok(word: u32) -> bool { word.count_ones() & 1 == 1 }`
in `packet_types.rs`, refactor `PacketHeader::decode` to call it (pure
refactor, no behavior change -- locked by the existing `decode` tests),
and call it for the Second_Header check. Do not re-derive the formula.

### 3.1 First_Header_Parity (bit 0)

- **Where:** `src/device/control_packets/reassembler.rs`, the
  `WaitingForStreamHeader` -> `Idle` transition (`feed_word`, currently
  reassembler.rs:98-112, which consumes the routing header raw via
  `word & 0x1F` etc.).
- **Mechanism:** the stream routing header is exactly what
  `PacketHeader::decode(word) -> (PacketHeader, parity_ok)` parses. The
  reassembler currently ignores `decode` and hand-extracts fields.
  Switch it to call `PacketHeader::decode`; on `parity_ok == false`
  return `ReassembleResult::HandlerError(PktHandlerError::
  FirstHeaderParity)`.
- **Reachability:** only when `Drop_Header=false` (the stream switch
  forwarded the routing header). When dropped, the handler never sees
  the header and the bit correctly cannot fire. Document this in code.

### 3.2 Second_Header_Parity (bit 1) -- tightening

- **Where:** `reassembler.rs`, opcode-header parse.
- **Condition:** odd-parity check on the 32-bit control-packet opcode
  header fails. Checked **before** structural decode (hardware validates
  header parity at ingress, before acting on opcode/length).
- **Behavior change:** structural parse rejections (invalid opcode,
  payload-length mismatch, read-with-payload) **no longer set bit 1**.
  They keep their existing `log` + `CtrlPacketAction::Error` path with
  no `pkt_handler_status` write. This is the only change to existing
  wired behavior and gets an explicit regression-locking test (5.2).

### 3.3 SLVERR_On_Access (bit 2) -- DEFERRED to successor plan

Not implemented by this plan. The original scoping assumed
`ControlPacketProcessor::process()` was the live register-access path
and that bit 2 was a small mirror of the parity path. Tracing the
runtime disproved both:

- `ControlPacketProcessor::process()` is **dead code at runtime** --
  invoked only by its own unit tests. The live path is
  `routing.rs::step_ctrl_packets` -> `reassembler.feed_word` ->
  `Complete(packet)` -> `packet_to_actions()` ->
  `CtrlPacketAction::{WriteRegister,ReadRegisters,Error}` ->
  `coordinator.rs` -> `DeviceState::write_tile_register`.
- `write_tile_register` returns `()` and **cannot fail**: an unmapped
  offset is silently stored in a `HashMap`. There is no
  address-decode / privilege model, so no faithful slave error arises.

Wiring bit 2 honestly means giving the runtime register-access path a
real error notion and threading it out through `write_tile_register`
plus the three `CtrlPacketAction` dispatch sites in `coordinator.rs`
(~298, ~361, ~848). That is a distinct subsystem-shaped change, not a
mirror of parity. Per the project rule against inventing simplified
approximations, it gets its own spec/plan as the **immediate successor**
to this one (Section 8). This plan deliberately leaves bit 2 unwired and
narrows the coverage annotation to name exactly that.

### 3.4 Tlast_Error (bit 3)

- **Where:** `reassembler.rs`, write-type packet completion.
- **Condition:** for Write / WriteIncr / BlockWrite, TLAST is in the
  wrong position -- **either** not asserted on the final declared beat
  **or** asserted early on a non-final beat ("missing or unexpected
  TLAST"). Read packets are header-only: no TLAST check.

## 4. Data flow / error channel

The reassembler's stringly `ReassembleResult::Error(String)` splits:

- **New** `ReassembleResult::HandlerError(PktHandlerError)` -- the
  status-bit cases (first/second parity, Tlast).
- **Existing** `ReassembleResult::Error(String)` -- structural
  rejections (log only, no bit; see 3.2).

`src/device/array/routing.rs::step_ctrl_packets` gets one private latch
helper, e.g. `fn latch_pkt_error(tile: &mut Tile, e: PktHandlerError)`
doing `tile.pkt_handler_status |= e.bit()`. The match becomes:

- `HandlerError(e)` -> `latch_pkt_error(tile, e)` + push
  `CtrlPacketAction::Error`.
- `Error(msg)` -> `log` + push `CtrlPacketAction::Error`, **no bit**
  (replaces the current `self.tiles[i].pkt_handler_status |= 0x2` at
  `routing.rs:363`, which today fires for every structural parse
  failure -- this removal IS the bit-1 tightening of 3.2).

One latch point, one bit map (`PktHandlerError::bit()`), one origin
(the reassembler) for this plan. The successor SLVERR plan adds a second
origin reaching the same `latch_pkt_error`.

Unchanged and explicitly relied upon:

- Write-1-to-clear: `state/effects.rs` already does
  `tile.pkt_handler_status &= !(value & 0xF)` for `0x3FF30`/`0xB0F30`
  -- covers all 4 bits.
- Read path: `tile/registers.rs` already returns
  `pkt_handler_status & 0xF` -- covers all 4 bits.

## 5. Testing

Validation tier is unit-level at the reassembler / processor / routing
seam. These are synthetic error injections that real hardware cannot
deterministically produce, so the bridge/HW path is **not** a meaningful
validation tier here -- validate at the realistic tier rather than
pretend HW emits a parity fault on demand. TDD: test first, watch it
fail, implement, watch it pass.

### 5.1 New positive-detection tests

- `reassembler.rs`: first-header parity fail -> `HandlerError(
  FirstHeaderParity)`; second-header parity fail -> `HandlerError(
  SecondHeaderParity)`; Tlast missing on final beat -> `HandlerError(
  Tlast)`; Tlast asserted early on non-final beat -> `HandlerError(
  Tlast)`; clean packet -> no error.
- After latching through `routing.rs`, the corresponding
  `pkt_handler_status` bit reads back via the register path.
- `registers.rs`: extend existing pkt_handler_status tests to cover
  read-back + write-1-to-clear for bits 0 and 3 individually (bit 1
  already covered; bit 2 deferred).
- `packet_types.rs`: the `decode` refactor is locked by existing
  `decode` tests -- no behavior change. Add one direct `odd_parity_ok`
  unit test (even-ones word -> false, odd-ones word -> true).

### 5.2 Regression-locking test (the tightening)

A structural parse error (e.g. payload-length mismatch) produces
`ReassembleResult::Error(_)` and **does not** set bit 1. This locks the
one behavior change in 3.2 against regression.

### 5.3 Gate

`cargo test --lib` green for xdna-emu and xdna-archspec.
`TMPDIR=/tmp/claude-1000` for sandbox-safe temp dirs. Zero coverage
artifact drift (Section 6).

## 6. Coverage lockstep (cross-task ordering)

Changing the verdict mechanically changes generated artifacts. Per the
Plan-3 staleness-gate discipline, regenerated docs commit **with** the
code that changes them, in the same task:

1. `src/device/tile/mod.rs` -- rewrite the `pkt_handler_status` doc
   block to the true AM025 map (`[0] First_Header_Parity_Error`,
   `[1] Second_Header_Parity_Error`, `[2] SLVERR_On_Access`,
   `[3] Tlast_Error`). The comment documents the hardware register, so
   it lists all four; annotate that bit 2 has no detecting path yet
   (successor plan).
2. `src/device/state/effects.rs:28-30` -- the comment currently reads
   "Tlast_Error / SLVERR_On_Access / Second_Header_Parity / ID_Parity"
   (wrong name + wrong order). Correct to the true AM025 map matching
   item 1. Behavior (`&= !(value & 0xF)`) is unchanged.
3. `crates/xdna-archspec/src/coverage/units.rs:177-180` -- the
   `control_packets` seed: verdict **stays** `Modeled{Partial{...}}`
   but `missing` narrows from
   `"Tlast/SLVERR/ID_Parity packet-handler sticky bits"` to exactly
   `"SLVERR_On_Access sticky bit -- needs a register-access-error
   model (successor plan)"`. Narrative updated: First/Second-header
   parity + Tlast sticky bits now have detecting paths +
   write-1-to-clear; SLVERR deferred (no faithful trigger until the
   runtime register-access path is repaired).
4. Regenerate: `cargo run -p xdna-archspec --example
   gen_coverage_artifacts`. Expected diffs:
   - `docs/coverage/aie2/subsystem-index.md` -- `control_packets` row
     narrative + `missing` text update; verdict stays
     `Modeled{Partial}`.
   - `docs/coverage/aie2/implementation-gaps.md` -- the
     `control_packets: PARTIAL ...` line **stays** (queue remains 5)
     but its text now names SLVERR specifically.
   - `docs/coverage/aie2/architecture-index.md` -- regenerate; changes
     only if a rolled-up category verdict moves (it should not, since
     the verdict is unchanged).
5. Zero-drift check after commit (regenerate again, `git diff` empty),
   exactly as Plan 3 did.

## 7. Files

- Create: `src/device/control_packets/status.rs` (+ `mod.rs`
  registration and `pub use`).
- Modify: `src/device/stream_switch/packet_types.rs` (add
  `odd_parity_ok`, refactor `decode` to use it),
  `src/device/control_packets/reassembler.rs` (parity + Tlast
  detection, `HandlerError` variant),
  `src/device/array/routing.rs` (latch helper + match rewrite),
  `src/device/tile/mod.rs` (comment), `src/device/state/effects.rs`
  (comment), `crates/xdna-archspec/src/coverage/units.rs` (seed
  narrative + narrowed `missing`).
- Regenerate: the three `docs/coverage/aie2/*.md` artifacts.
- Tests: alongside each modified module (reassembler / registers /
  packet_types), per existing patterns. `processor.rs` is untouched
  (dead at runtime; SLVERR successor plan owns it).

## 8. Out of scope / non-goals

- No regdb-driven bit extraction (stable 4-bit register; centralized
  literals + AM025 cite chosen deliberately).
- No HW/bridge validation tier (synthetic faults; see Section 5).
- No change to the structural-parse-rejection logging behavior beyond
  decoupling it from bit 1.
- No broader control-packet feature work (response routing, opcode
  coverage) -- this closes exactly the three faithfully-detectable
  status bits.

## 9. Successor plan (committed)

SLVERR_On_Access (bit 2) is the **immediate next plan** after this one.
It must repair the runtime register-access path so a slave error is a
real, faithful event rather than a fabricated one:

- Give `DeviceState::write_tile_register` (and the read path behind
  `CtrlPacketAction::ReadRegisters` / `handle_read_registers`) a
  fallible result expressing address-decode / access failure.
- Thread that result through the three `CtrlPacketAction` dispatch
  sites in `coordinator.rs` (~298, ~361, ~848) to the existing
  `latch_pkt_error` helper.
- Add the `Slverr` variant to `PktHandlerError` (the exhaustive
  no-`_` match will force every site to handle it -- the intended
  trip-wire).
- Decide the faithful definition of an emulator slave error (likely:
  offset not a decodable register for the tile type, grounded in the
  regdb register map rather than an ad-hoc list).
- Flip the `control_packets` coverage verdict to `Modeled{Full}` and
  drop it from `implementation-gaps.md` (queue 5 -> 4) in lockstep.

This is recorded so the deferral is a tracked commitment, not a
silently dropped requirement.
