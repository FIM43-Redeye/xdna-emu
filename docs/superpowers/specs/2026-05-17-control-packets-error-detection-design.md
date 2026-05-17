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
inherited that error. This work wires the three genuinely unwired bits
(0, 2, 3), tightens bit 1 to true-parity-only, and corrects the misnomer
in both the comment and the coverage string so the gap closes against
reality.

## 1. Scope

In scope:

- Detecting paths + sticky-bit latching for First_Header_Parity (0),
  SLVERR_On_Access (2), Tlast_Error (3).
- Tightening Second_Header_Parity (1) to fire only on a genuine
  opcode-header odd-parity failure (today it fires on every header parse
  failure, including structural ones).
- Correcting the `ID_Parity_Error` misnomer in `tile/mod.rs` and the
  `units.rs` coverage seed.
- Lockstep regeneration of the three generated coverage artifacts.

Out of scope: no change to write-1-to-clear (`effects.rs` already covers
all 4 bits) or the register read path (`registers.rs` already returns
`& 0xF`). Both were built generic; only detection was missing. No new
regdb-driven extraction -- bit positions are centralized literals with an
AM025 citation.

## 2. New unit: `PktHandlerError`

New file `src/device/control_packets/status.rs`:

```rust
/// The four Tile_Control_Packet_Handler_Status sticky-error conditions.
/// Bit positions per AM025 Tile_Control_Packet_Handler_Status
/// (regdb fields First_Header_Parity_Error / Second_Header_Parity_Error /
/// SLVERR_On_Access / Tlast_Error). This is the single source of truth
/// for the bit map; no other site names these positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PktHandlerError {
    FirstHeaderParity,  // bit 0
    SecondHeaderParity, // bit 1
    Slverr,             // bit 2
    Tlast,              // bit 3
}

impl PktHandlerError {
    pub fn bit(self) -> u32 {
        match self {
            PktHandlerError::FirstHeaderParity => 0x1,
            PktHandlerError::SecondHeaderParity => 0x2,
            PktHandlerError::Slverr => 0x4,
            PktHandlerError::Tlast => 0x8,
        }
    }
}
```

Exhaustive match, no `_` arm (project convention -- a new variant must
force a compile error here).

## 3. Detection semantics

Parity is computed by **reusing the existing odd-parity helper used by
`PacketHeader::decode` in `src/device/.../packet_types.rs`**. Do not
re-derive a parity formula; locate the existing one and call it. (If it
is private, lift it to a shared helper rather than duplicate.)

### 3.1 First_Header_Parity (bit 0)

- **Where:** `src/device/control_packets/reassembler.rs`, the
  `WaitingForStreamHeader` -> `Idle` transition.
- **Condition:** odd-parity check on the 32-bit stream routing header
  word fails.
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

### 3.3 SLVERR_On_Access (bit 2)

- **Where:** `src/device/control_packets/processor.rs` register-access
  path (`RegisterAccess::write_register` / `read_register`).
- **Condition:** any such call returns `Err` during packet processing --
  the emulator's analog of an AXI slave error (unmapped / decode-failed
  / denied access).
- **Origin differs:** this is the processor, not the reassembler.
  The processor surfaces "an access error occurred" up to `routing.rs`,
  which latches `PktHandlerError::Slverr` through the same latch helper
  (Section 4). The processor stays free of tile-status state.

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
  `routing.rs:363`).

SLVERR (Section 3.3) reaches the same `latch_pkt_error` from the
processor-result path. One latch point, one bit map (`PktHandlerError::
bit()`), two origins (reassembler, processor).

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
- `processor.rs` / `routing.rs`: a `RegisterAccess` `Err` during
  processing latches bit 2.
- After latching through `routing.rs`, the corresponding
  `pkt_handler_status` bit reads back via the register path.
- `effects.rs` / `registers.rs`: extend existing pkt_handler_status
  tests to cover read-back + write-1-to-clear for bits 0, 2, 3
  individually (bit 1 already covered).

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
   `[3] Tlast_Error`); fix the "we OR a bit in when the reassembler
   observes" sentence (SLVERR originates in the processor).
2. `crates/xdna-archspec/src/coverage/units.rs:177-180` -- the
   `control_packets` seed: verdict
   `Modeled{Partial{missing:"Tlast/SLVERR/ID_Parity..."}}` ->
   `Modeled{completeness: Full}`; narrative loses the "Second_Header
   _Parity_Error wired, Tlast/SLVERR/ID_Parity not (no detecting path)"
   sentence, replaced with: all four `Tile_Control_Packet_Handler_Status`
   sticky bits have detecting paths + write-1-to-clear.
3. Regenerate: `cargo run -p xdna-archspec --example
   gen_coverage_artifacts`. Expected diffs:
   - `docs/coverage/aie2/subsystem-index.md` -- `control_packets`
     row -> `Modeled { completeness: Full }`, drift recomputed.
   - `docs/coverage/aie2/implementation-gaps.md` -- the
     `control_packets: PARTIAL ...` line drops off (queue 5 -> 4).
   - `docs/coverage/aie2/architecture-index.md` -- regenerate; changes
     only if a rolled-up category verdict moves.
4. Zero-drift check after commit (regenerate again, `git diff` empty),
   exactly as Plan 3 did.

## 7. Files

- Create: `src/device/control_packets/status.rs` (+ `mod.rs`
  registration).
- Modify: `src/device/control_packets/reassembler.rs` (parity + Tlast
  detection, `HandlerError` variant), `processor.rs` (surface access
  error), `src/device/array/routing.rs` (latch helper + match rewrite),
  `src/device/tile/mod.rs` (comment), `crates/xdna-archspec/src/coverage
  /units.rs` (seed).
- Locate + reuse: existing odd-parity helper near `PacketHeader::decode`.
- Regenerate: the three `docs/coverage/aie2/*.md` artifacts.
- Tests: alongside each modified module (reassembler/processor/registers/
  effects), per existing patterns.

## 8. Out of scope / non-goals

- No regdb-driven bit extraction (stable 4-bit register; centralized
  literals + AM025 cite chosen deliberately).
- No HW/bridge validation tier (synthetic faults; see Section 5).
- No change to the structural-parse-rejection logging behavior beyond
  decoupling it from bit 1.
- No broader control-packet feature work (response routing, opcode
  coverage) -- this closes exactly the status-bit gap.
