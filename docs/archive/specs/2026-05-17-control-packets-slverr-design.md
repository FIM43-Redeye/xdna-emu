# Control-Packet Handler SLVERR + Sticky-Continue Fidelity -- Design

**Status:** approved (brainstorming, 2026-05-17)
**Closes:** `control_packets` PARTIAL implementation gap (coverage spine)
**Outcome:** `control_packets` -> `Modeled { completeness: Full }`, queue 5 -> 4
**Predecessor:** `2026-05-17-control-packets-error-detection-design.md`
(the 3-bit plan; this is its committed immediate successor, recorded in
that spec's Section 9)

## 0. Problem

The 3-bit plan wired First_Header_Parity (0), Second_Header_Parity (1),
and Tlast_Error (3) of `Tile_Control_Packet_Handler_Status` (compute
`0x3FF30`, memtile `0xB0F30`). It deliberately deferred
**SLVERR_On_Access (bit 2)** because tracing the live path showed its
original scoping premise was false: `ControlPacketProcessor::process()`
is dead code at runtime and `DeviceState::write_tile_register` returns
`()` and cannot fail, so no faithful slave error arose. `control_packets`
was left at an honest `Modeled{Partial}` with `missing` naming exactly
SLVERR.

This plan wires SLVERR faithfully and, in the same lockstep, corrects a
fidelity bug the 3-bit plan's `fatal_errors` routing introduced for the
three shipped bits. It closes `control_packets` to `Full`.

### 0.1 Derivation (what the toolchain says)

Probed aie-rt (authoritative HAL) and AM025 extracts. Findings, cited:

- **Bit map confirmed.** aie-rt
  `XAIEGBL_CORE_VALUE_TILCTRLPKTHANSTA(Tlast_Error, SLVERR_On_Access,
  Second_Header_Parity_Error, First_Header_Parity_Error)` =
  `(Tlast<<3)+(SLVERR<<2)+(Second<<1)+(First<<0)`
  (`/home/triple/npu-work/aie-rt/driver/src/global/xaiegbl_params.h:7761`).
  All four are `wtc` (write-1-to-clear) sticky bits, reset 0
  (AM025 `docs/xdna/am025-compact/core_module/tile_control.txt:13-17`,
  `memory_tile_module/tile_control.txt:13-17`,
  `pl_module/control_packet_handler.txt:7-11`). Our shipped order is
  correct -- no bit-map change.

- **SLVERR trigger = AXI unmapped-register address decode.** AM025
  `AIE_AXIMM_Config @ 0x1E020` bit 2 `SLVERR_Block`:
  *"1: Block returning of SLVERR when accessing unmapped registers"*
  (`docs/xdna/am025-compact/noc_module/aie_aximm_config.txt:9`; bit 3
  `DECERR_Block` "accessing non-existent tiles" is the adjacent line 8).
  SLVERR is the slave-error response to an *unmapped/undecoded register
  offset within an existing tile* -- distinct from DECERR (no tile at
  the address), which the emulator never produces because control
  packets are always routed to real tiles. Explicitly **not** a
  privilege/protection violation and **not** a write to a read-only
  register.

- **Handler is poll-only sticky-and-continue.** Neither aie-rt (no live
  control-packet handler code) nor AM025 (declarative `wtc` bit, no
  trigger FSM, no interrupt wiring) ties any of the four conditions to
  an interrupt, error broadcast, or handler halt. The faithful model:
  latch the sticky bit, continue processing the next packet, firmware
  polls. No abort.

- **Shipped 3-bit code diverges (the reopened bug).** Our shipped path
  is `reassembler HandlerError(e)` -> `routing.rs::step_ctrl_packets`
  latches the bit **and** pushes `CtrlPacketAction::Error` ->
  `coordinator.rs` Error arm `self.device.array.fatal_errors.push(msg)`
  (`coordinator.rs:305`, `:368`, `:855`) -> next
  `drain_fatal_errors()` (`coordinator.rs:283`, `:833`) sets
  `EngineStatus::Error` and returns. `EngineStatus::Error` is terminal
  (`coordinator.rs:536`, `:1407`). So a poll-only status condition
  currently **aborts the engine**. Fixing SLVERR's semantics while
  leaving its three siblings unfaithful would be incoherent -- all four
  converge on one latch and one Error channel -- so this plan corrects
  all four together.

## 1. Scope

In scope:

- Add `PktHandlerError::Slverr` (bit `0x4`) -- the no-`_` exhaustive
  match trip-wire planted by the 3-bit plan forces the new arm.
- Faithful SLVERR trigger: a control-packet-handler-initiated register
  access (read or write) whose offset classifies as
  `SubsystemKind::Unknown` for the tile kind.
- A shared `dispatch_ctrl_action` helper replacing the three byte-
  identical `CtrlPacketAction` match blocks in `coordinator.rs`
  (~298, ~361, ~848); the SLVERR decode-gate lives in that one place.
- **Fidelity correction for all four bits:** handler-status-bit
  conditions latch the sticky bit, log, and **continue** -- they no
  longer push `CtrlPacketAction::Error` (the engine-fatal channel).
  Includes correcting any shipped test that asserts engine-abort on
  parity/Tlast.
- Coverage lockstep: `control_packets` -> `Modeled{Full}`, dropped from
  `implementation-gaps.md` (queue 5 -> 4), with a forward-link to the
  `noc`-gated SLVERR_Block refinement (Section 10).

Out of scope:

- **`SLVERR_Block` modeling** -- tracked goal, NoC-gated (Section 10).
- **Structural-rejection fatality.** `ReassembleResult::Error(msg)`
  (invalid opcode, payload-length mismatch) is *not* an AM025 status-bit
  condition. Its current `Error -> fatal_errors -> abort` behavior is a
  separate design question; this plan deliberately does not touch it
  (no scope creep into structural-rejection policy).
- No fallible `write_tile_register` (decided: predicate at the dispatch
  boundary instead -- the bus has 30+ non-ctrl-pkt callers and a CDO
  write to an unmapped offset is not the control-packet handler's
  SLVERR).
- No regdb offset-set extraction (decided: `SubsystemKind::Unknown`
  classification is the already-toolchain-derived signal).
- No RO-write SLVERR (derivation: AIE RO writes silently drop; SLVERR is
  unmapped-decode only).
- No HW/bridge validation tier -- SLVERR is not deterministically
  triggerable on silicon (same class as parity faults; a real binary
  never sends a control packet to an unmapped address). Validate at the
  realistic unit tier.

## 2. `PktHandlerError::Slverr`

`src/device/control_packets/status.rs`: add the variant and its bit.

```rust
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

Exhaustive, no `_` arm (unchanged convention). The doc comment's
"SLVERR_On_Access (bit 0x4) is intentionally absent ..." paragraph is
replaced with the faithful trigger and a cite to
`xaiegbl_params.h:7761` / `aie_aximm_config.txt:2-3`.

## 3. Faithful slave-error definition

A control-packet-handler-initiated register access (read **or** write)
is a slave error iff the target offset classifies as
`SubsystemKind::Unknown` for the tile kind, via the existing
`subsystem_from_offset(offset, tile_kind_from_row(row))`
(`src/device/registers.rs:410`). This is the AXI SLVERR case -- the
tile exists (the packet is routed to a real `(col,row)`) but the offset
maps to no register (AM025 `SLVERR_Block`, "unmapped registers"); it is
not DECERR (no tile at the address), which never arises here. The
classifier is derived from the
AM025/aie-rt subsystem range constants and already returns `Unknown`
only for genuinely undecoded space (verified: compute `0x10800`,
`0x1F200`, `0x50000` -> `Unknown`; `0x10000`-class data memory, DMA-BD
strides, lock/stream-switch ranges classify as their real subsystem,
never `Unknown`). No new extraction, and it inherently cannot
false-positive on legitimate memory or register-array writes.

## 4. Unified faithful semantic for all four bits

Every handler-status-bit condition -- First/Second-header parity, Tlast
(reassembler origin) and SLVERR (coordinator origin) -- resolves to the
identical faithful behavior:

1. Resolve the tile.
2. `tile.pkt_handler_status |= PktHandlerError::<variant>.bit()`.
3. `log::error!(...)` (diagnostic).
4. **Continue** -- the offending access is not performed (write
   skipped, so the raw `registers.insert` in `write_register` never
   runs because the gate precedes `write_tile_register`; read response
   not queued), and the handler proceeds to the next packet.

None of the four push `CtrlPacketAction::Error`. Concrete changes:

- **`src/device/array/routing.rs::step_ctrl_packets`, `HandlerError(e)`
  arm:** keep `Self::latch_pkt_error(&mut self.tiles[i], e)` + the
  `log::error!`; **remove**
  `self.pending_ctrl_actions.push(CtrlPacketAction::Error(format!("{:?}", e)))`.
  This is the shipped-code fidelity correction (parity/Tlast no longer
  engine-fatal).
- **SLVERR in `dispatch_ctrl_action` (Section 6):** call
  `latch_ctrl_slverr(col, row)` + `log::error!`; no Error push; access
  skipped.
- **`Error(msg)` structural-rejection arm: unchanged** (out of scope,
  Section 1).

### 4.1 Shipped-test correction (TDD)

The 3-bit plan's tests may assert engine-abort / `EngineStatus::Error` /
`fatal_errors` non-empty on a parity or Tlast injection. Those encoded
the now-corrected approximation. Audit every test touching parity/Tlast
handler errors; rewrite each to assert: the corresponding
`pkt_handler_status` bit is set, *and* the engine has **not** entered
`EngineStatus::Error` from the handler condition (it continues). The
corrected sticky-continue behavior is the spec; tests assert the spec.

## 5. Decode predicate + read-range

Pure, non-mutating `DeviceState` methods (small new
`src/device/state/ctrl_access.rs`, or co-located in `dispatch.rs`):

```rust
/// A control-packet access whose offset classifies as
/// SubsystemKind::Unknown is the faithful AXI SLVERR case: the tile
/// exists but the offset maps to no register. Distinct from DECERR
/// (no tile at the address), which never arises -- packets route to
/// real tiles. Cite: xaiegbl_params.h:7761 (bit map),
/// aie_aximm_config.txt:9 SLVERR_Block (trigger).
pub fn ctrl_pkt_offset_decodes(&self, row: u8, offset: u32) -> bool {
    subsystem_from_offset(offset, tile_kind_from_row(row)) != SubsystemKind::Unknown
}

/// A read of `count` consecutive registers from `offset` decodes iff
/// every beat decodes. Any undecoded beat -> SLVERR, whole response
/// suppressed. `count == 0` is a header-only read (no register access,
/// mirroring `handle_read_registers`) and is vacuously decodable --
/// it cannot raise SLVERR.
pub fn ctrl_pkt_read_range_decodes(&self, row: u8, offset: u32, count: u8) -> bool {
    (0..count as u32).all(|i| self.ctrl_pkt_offset_decodes(row, offset + i * 4))
}
```

Classification is tile-kind (`row`) + `offset` only; `col` is
irrelevant to address decode.

`latch_ctrl_slverr(&mut self, col, row)` resolves the tile and ORs in
`PktHandlerError::Slverr.bit()` -- the SLVERR coordinator-origin
equivalent of the reassembler-origin `latch_pkt_error`. Both paths end
at `pkt_handler_status |= PktHandlerError::*.bit()`: one bit map, two
origins, the single-latch invariant the 3-bit spec Section 4
anticipated ("the successor SLVERR plan adds a second origin reaching
the same latch").

## 6. Shared dispatch helper (the DRY)

The three byte-identical match blocks (`coordinator.rs` ~298, ~361,
~848) collapse into one engine method that owns the SLVERR gate:

```rust
fn dispatch_ctrl_action(&mut self, action: CtrlPacketAction) {
    use crate::device::tile::CtrlPacketAction;
    match action {
        CtrlPacketAction::WriteRegister { col, row, offset, value } => {
            if self.device.ctrl_pkt_offset_decodes(row, offset) {
                self.device.write_tile_register(col, row, offset, value);
            } else {
                log::error!(
                    "Tile ({},{}) ctrl_pkt SLVERR: write to undecoded offset 0x{:05X} \
                     (sets Control_Packet_Handler_Status bit 0x4)",
                    col, row, offset
                );
                self.device.latch_ctrl_slverr(col, row);
            }
        }
        CtrlPacketAction::ReadRegisters { col, row, offset, count, response_id } => {
            if self.device.ctrl_pkt_read_range_decodes(row, offset, count) {
                self.device.array.handle_read_registers(col, row, offset, count, response_id);
            } else {
                log::error!(
                    "Tile ({},{}) ctrl_pkt SLVERR: read of {} regs from undecoded \
                     offset 0x{:05X} (sets Control_Packet_Handler_Status bit 0x4)",
                    col, row, count, offset
                );
                self.device.latch_ctrl_slverr(col, row);
            }
        }
        CtrlPacketAction::Error(msg) => {
            self.device.array.fatal_errors.push(msg);
        }
    }
}
```

Each of the three sites becomes
`for action in ctrl_actions { self.dispatch_ctrl_action(action); }`.
Exact method placement/signature (the three sites sit in `impl` blocks
on the engine type owning `self.device`) is settled in the plan; the
helper must be reachable from all three with no behavior change to the
`Error` arm.

## 7. Testing

Unit tier at the dispatch / routing / status seam. TDD: test first,
watch it fail, implement, watch it pass.

### 7.1 SLVERR positive detection

- ctrl `WriteRegister` action to an `Unknown` offset (compute
  `0x1F200`) -> `pkt_handler_status & 0x4 != 0`; offset **absent** from
  `tile.registers` (write suppressed); engine **continues** (not
  `EngineStatus::Error`).
- ctrl `ReadRegisters` action with an `Unknown` offset -> bit 2 set; no
  response words queued; engine continues.
- `PktHandlerError::Slverr.bit() == 0x4`; exhaustive match compiles
  (trip-wire satisfied).
- Write-1-to-clear bit 2 via `effects.rs` (already generic) + read-back
  via `registers.rs`.

### 7.2 False-positive guard (load-bearing)

ctrl `WriteRegister` to a valid data-memory offset **and** a DMA-BD
register-array offset -> bit 2 **not** set, write applied. Locks the
"`Unknown` excludes memory/arrays" claim of Section 3 against
regression.

### 7.3 Sticky-continue regression lock (the §4 fidelity fix)

A parity injection and a Tlast injection each set their bit **and** the
engine does **not** enter `EngineStatus::Error` from the handler
condition (it continues to process subsequent packets). This locks the
shipped-code correction and prevents regressing to fatal-abort. Replaces
/ corrects any shipped test asserting the old abort behavior (4.1).

### 7.4 Gate

`cargo test --lib` green for xdna-emu and xdna-archspec.
`TMPDIR=/tmp/claude-1000` for sandbox-safe temp dirs. Zero coverage
artifact drift (Section 8).

## 8. Coverage lockstep (cross-task ordering)

Per the Plan-3 staleness-gate discipline, regenerated docs commit
**with** the code that changes them, in the same task:

1. `crates/xdna-archspec/src/coverage/units.rs` -- the
   `control_packets` seed: verdict `Modeled{Partial{...}}` ->
   `Modeled{Full}`; drop `missing`. Narrative: all four
   `Tile_Control_Packet_Handler_Status` sticky bits now have faithful
   detecting paths (First/Second-header parity + Tlast via the
   reassembler; SLVERR via undecoded-address decode at the dispatch
   boundary) with poll-only sticky-continue semantics matching
   aie-rt/AM025; the `SLVERR_Block` config-suppression refinement is a
   tracked `noc`-gated goal (Section 10), cross-referenced so it is
   discoverable from the `noc` gap rather than misattributed here.
2. `src/device/tile/mod.rs` + `src/device/state/effects.rs` doc blocks
   -- drop the "bit 2 has no detecting path yet (successor plan)"
   annotation; state all four bits are faithfully detected and
   poll-only sticky. `effects.rs` behavior (`&= !(value & 0xF)`)
   unchanged.
3. Regenerate: `cargo run -p xdna-archspec --example
   gen_coverage_artifacts`. Expected diffs:
   - `docs/coverage/aie2/subsystem-index.md` -- `control_packets`
     verdict `Modeled{Partial}` -> `Modeled{Full}`, `missing` cleared,
     narrative updated; `noc` row gains the forward-referenced
     SLVERR_Block dependency note.
   - `docs/coverage/aie2/implementation-gaps.md` -- the
     `control_packets: PARTIAL ...` line **removed**, queue 5 -> 4.
   - `docs/coverage/aie2/architecture-index.md` -- regenerate; the
     rolled-up category verdict for the control-packets domain moves to
     fully modeled. Accept the derived diff.
4. Zero-drift check after commit (regenerate again, `git diff` empty),
   exactly as Plan 3 did.

## 9. Files

- Modify: `src/device/control_packets/status.rs` (add `Slverr`, bit
  `0x4`, doc), `src/device/state/ctrl_access.rs` (new -- predicate,
  read-range, `latch_ctrl_slverr`) or co-located in
  `src/device/state/dispatch.rs`, `src/interpreter/engine/coordinator.rs`
  (extract `dispatch_ctrl_action`, replace 3 sites),
  `src/device/array/routing.rs` (`HandlerError` arm: drop the Error
  push), `src/device/tile/mod.rs` + `src/device/state/effects.rs`
  (doc-block update), `crates/xdna-archspec/src/coverage/units.rs`
  (verdict -> Full + forward-link).
- Regenerate: the three `docs/coverage/aie2/*.md` artifacts.
- Tests: alongside coordinator / routing / status / effects /
  registers per existing patterns; shipped parity/Tlast tests corrected
  per 4.1 / 7.3.
- Untouched: `src/device/control_packets/processor.rs` -- still dead at
  runtime. SLVERR rides the live coordinator path, vindicating the
  3-bit plan's deferral reasoning; `process()` is not revived.

## 10. SLVERR_Block: tracked goal, NoC-gated

AM025 `AIE_AXIMM_Config @ 0x1E020` bit 2 `SLVERR_Block`
(`docs/xdna/am025-compact/noc_module/aie_aximm_config.txt:2-3`):
*"1: Block returning of SLVERR when accessing unmapped registers."*
Full fidelity means latching SLVERR only when `SLVERR_Block == 0`.
`AIE_AXIMM_Config` is a NoC-module register; honoring it requires the
NoC config path to be plumbed, which is `noc`-subsystem work (`noc` is
a STUB in the gap queue). This is therefore a **tracked, forward-linked
commitment**, not a silent simplification -- the same disciplined
deferral pattern by which SLVERR itself was tracked in the 3-bit spec's
Section 9.

The reset default is `SLVERR_Block == 0` (SLVERR enabled), so the
implemented behavior is faithful for **every observed binary**; the only
unmodeled path is the unobserved `SLVERR_Block == 1` suppression case,
which lives behind an unplumbed NoC register. `control_packets ->
Modeled{Full}` is honest because the control-packet *handler* is fully
modeled; the SLVERR_Block dependency is correctly attributed to the
`noc` gap via the cross-reference in Section 8, not misattributed to
the handler.

When `noc` closes (or `AIE_AXIMM_Config` is otherwise plumbed): read
`SLVERR_Block` in `ctrl_pkt_offset_decodes` / `ctrl_pkt_read_range_
decodes` and suppress the latch when set. Recorded here so the
deferral is a tracked obligation discoverable from both this spec and
the `noc` coverage gap.

## 11. Non-goals

- No `SLVERR_Block` modeling now (Section 10 -- tracked, NoC-gated).
- No change to structural-rejection (`ReassembleResult::Error`)
  fatality (Section 1 -- separate question).
- No fallible `write_tile_register`; no regdb offset-set extraction;
  no RO-write SLVERR; no HW/bridge tier; `processor.rs` not revived.
- No broader control-packet feature work (response routing, opcode
  coverage) -- this closes exactly the four faithfully-detectable
  status bits with their faithful poll-only sticky-continue semantics.
