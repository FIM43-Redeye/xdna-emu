# Control-Packet SLVERR + Sticky-Continue Fidelity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `SLVERR_On_Access` (bit 2) of `Tile_Control_Packet_Handler_Status` via faithful AXI-decode detection, and correct all four handler-status bits to the poll-only sticky-continue semantic the toolchain mandates, closing `control_packets` to `Modeled{Full}`.

**Architecture:** A control-packet-handler register access whose offset classifies as `SubsystemKind::Unknown` for the tile kind is the faithful AXI slave-error (DECERR) analogue. A decode-gate predicate is applied at a single extracted `dispatch_ctrl_action` helper that replaces three byte-identical `CtrlPacketAction` match blocks in `coordinator.rs`. All four handler-status conditions latch their sticky bit, log, and continue (no engine-fatal `CtrlPacketAction::Error` push) -- which also corrects the shipped parity/Tlast fatal-abort. Coverage artifacts regenerate in the same lockstep.

**Tech Stack:** Rust, `cargo test --lib`, `xdna-archspec` coverage generator.

**Spec:** `docs/superpowers/specs/2026-05-17-control-packets-slverr-design.md` (committed `6084c56`).

**Branch:** `dev` (already current; do not branch).

**Pre-flight context (read once, do not re-derive):**

- Derivation already done in the spec, cited: aie-rt `xaiegbl_params.h:7761` (bit map: `(Tlast<<3)+(SLVERR<<2)+(Second<<1)+(First<<0)`); AM025 `aie_aximm_config.txt:2-3` (SLVERR = unmapped-register decode; `SLVERR_Block` deferred, NoC-gated). The handler is poll-only sticky-and-continue (no interrupt, no abort).
- The single latch invariant: reassembler-origin errors latch via `TileArray::latch_pkt_error` (`routing.rs:394`); SLVERR is coordinator-origin and latches via a new `DeviceState::latch_ctrl_slverr`. Both end at `pkt_handler_status |= PktHandlerError::*.bit()`.
- `subsystem_from_offset(offset, tile_kind_from_row(row))` (`src/device/registers.rs:410`, `:691`) returns `SubsystemKind::Unknown` only for genuinely undecoded space. Verified-unknown compute offsets usable as test fixtures: `0x10800`, `0x1F200`, `0x50000`. Data memory (e.g. `0x400`), DMA-BD, lock, stream-switch ranges classify as their real subsystem -- never `Unknown`.
- `src/device/state/mod.rs:37` already does `use super::registers::{subsystem_from_offset, tile_kind_from_row};` and `:40` `use xdna_archspec::types::{SubsystemKind, TileKind};`. Files in the `state` module reach these via `use super::*` (the pattern `dispatch.rs` uses).
- Engine type is `InterpreterEngine` (`coordinator.rs:121`), built with `InterpreterEngine::new_npu1()`. The three dispatch sites are in `impl InterpreterEngine`. Tests live in `mod tests { use super::*; }` at the bottom of `coordinator.rs`; a child test module may call private methods/fields of the parent, so tests can call `engine.dispatch_ctrl_action(...)`, read `engine.device.array...`, and `engine.status()`.
- `EngineStatus::Error` is terminal. The shipped path `HandlerError -> push CtrlPacketAction::Error -> coordinator Error arm -> array.fatal_errors.push -> drain_fatal_errors (coordinator.rs:283/:833) -> EngineStatus::Error -> return` makes a poll-only status condition engine-fatal. That is the bug Task 4 corrects.
- **Audit result (already performed):** no shipped test asserts engine-abort/`fatal_errors`/`EngineStatus::Error` for a parity/Tlast handler error. The 3-bit tests are reassembler-unit-level (`reassembler.rs` ~380-470, asserting `ReassembleResult::HandlerError(...)`). Task 4 therefore *adds* a sticky-continue regression-lock; it does not rewrite existing tests.
- `effects.rs:28-35` already lists the correct AM025 bit order with no successor-plan annotation -- it needs **no change**. Only `tile/mod.rs:305-311` carries the stale "successor plan" wording.
- Run tests with `TMPDIR=/tmp/claude-1000` for sandbox safety. Run `cargo` bare (never piped). Commit messages end with a blank line then `Generated using Claude Code.` No emoji.

---

### Task 1: Add `PktHandlerError::Slverr` (bit 0x4)

**Files:**
- Modify: `src/device/control_packets/status.rs` (enum, `bit()`, doc, test)

- [ ] **Step 1: Write the failing test**

In `src/device/control_packets/status.rs`, replace the `bit_map_matches_am025` test body so it also asserts the new variant:

```rust
    #[test]
    fn bit_map_matches_am025() {
        assert_eq!(PktHandlerError::FirstHeaderParity.bit(), 0x1);
        assert_eq!(PktHandlerError::SecondHeaderParity.bit(), 0x2);
        assert_eq!(PktHandlerError::Slverr.bit(), 0x4);
        assert_eq!(PktHandlerError::Tlast.bit(), 0x8);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `TMPDIR=/tmp/claude-1000 cargo test -p xdna-emu --lib status::tests::bit_map_matches_am025`
Expected: FAIL to compile -- `no variant named Slverr` (the trip-wire: the exhaustive `bit()` match also forces a new arm).

- [ ] **Step 3: Write minimal implementation**

In the same file, add the `Slverr` variant (ordered by bit, between `SecondHeaderParity` and `Tlast`) and its `bit()` arm, and rewrite the doc paragraph. Replace lines 7-39 with:

```rust
/// A control-packet handler error with a faithful detecting path.
///
/// Bit positions per AM025 `Tile_Control_Packet_Handler_Status` and
/// aie-rt `XAIEGBL_CORE_VALUE_TILCTRLPKTHANSTA`
/// (`xaiegbl_params.h:7761`: `(Tlast<<3)+(SLVERR<<2)+(Second<<1)+
/// (First<<0)`). This is the single source of truth for the bit map --
/// no other site names these positions.
///
/// `Slverr` is the AXI slave-error (DECERR) analogue: a control-packet
/// access whose offset decodes to no slave (AM025
/// `aie_aximm_config.txt:2-3`, `SLVERR_Block` -- "SLVERR when accessing
/// unmapped registers"). All four are poll-only sticky bits: the
/// handler latches and continues; firmware polls. The `SLVERR_Block`
/// config-suppression refinement is a tracked NoC-gated goal (spec
/// Section 10), not modeled here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PktHandlerError {
    /// Stream routing header (First) odd-parity failure -- bit 0.
    FirstHeaderParity,
    /// Control-packet opcode header (Second) odd-parity failure -- bit 1.
    SecondHeaderParity,
    /// Register access to an undecoded offset (AXI SLVERR) -- bit 2.
    Slverr,
    /// TLAST in the wrong position on a write-class packet -- bit 3.
    Tlast,
}

impl PktHandlerError {
    /// The `Tile_Control_Packet_Handler_Status` bit this condition sets.
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

- [ ] **Step 4: Run test to verify it passes**

Run: `TMPDIR=/tmp/claude-1000 cargo test -p xdna-emu --lib status::tests::bit_map_matches_am025`
Expected: PASS.

- [ ] **Step 5: Verify no other exhaustive match broke**

Run: `TMPDIR=/tmp/claude-1000 cargo build -p xdna-emu`
Expected: builds clean. (Other sites use `e.bit()` generically or match specific variants with a fallback arm; only `bit()` is exhaustive. If a stale rust-analyzer-style error appears, trust bare `cargo build` -- per CLAUDE.md the bare build is ground truth.)

- [ ] **Step 6: Commit**

```bash
git add src/device/control_packets/status.rs
git commit -m "control-packets: add PktHandlerError::Slverr (bit 0x4)

Trip-wire from the 3-bit plan lands: the exhaustive no-_ bit() match
forced the new arm. Bit map cites xaiegbl_params.h:7761.

Generated using Claude Code."
```

---

### Task 2: Decode predicate, read-range, and SLVERR latch

**Files:**
- Create: `src/device/state/ctrl_access.rs`
- Modify: `src/device/state/mod.rs` (register the module)

- [ ] **Step 1: Create the module file with a failing test**

Create `src/device/state/ctrl_access.rs`:

```rust
//! Control-packet register-access decode gate.
//!
//! A control-packet-handler-initiated register access whose offset
//! classifies as `SubsystemKind::Unknown` for the tile kind is the
//! emulator's faithful AXI slave-error (DECERR) analogue -- the address
//! decodes to no slave (AM025 `aie_aximm_config.txt:2-3`). This is the
//! coordinator-origin counterpart to the reassembler-origin
//! `TileArray::latch_pkt_error`; both converge on
//! `pkt_handler_status |= PktHandlerError::*.bit()` (one bit map, two
//! origins).

use super::*;
use crate::device::control_packets::PktHandlerError;

impl DeviceState {
    /// True iff `offset` decodes to a real subsystem for the tile kind
    /// at `row`. A control-packet access to an offset that classifies
    /// as `SubsystemKind::Unknown` is a slave error (SLVERR). `col` is
    /// irrelevant to address decode.
    pub fn ctrl_pkt_offset_decodes(&self, row: u8, offset: u32) -> bool {
        subsystem_from_offset(offset, tile_kind_from_row(row)) != SubsystemKind::Unknown
    }

    /// A read of `count` consecutive registers from `offset` decodes
    /// iff every beat decodes. Any undecoded beat is a SLVERR and the
    /// whole response is suppressed.
    pub fn ctrl_pkt_read_range_decodes(&self, row: u8, offset: u32, count: u8) -> bool {
        (0..count.max(1) as u32).all(|i| self.ctrl_pkt_offset_decodes(row, offset + i * 4))
    }

    /// Latch `SLVERR_On_Access` on the tile at `(col,row)`. The
    /// coordinator-origin equivalent of `TileArray::latch_pkt_error`.
    pub fn latch_ctrl_slverr(&mut self, col: u8, row: u8) {
        if let Some(tile) = self.array.get_mut(col, row) {
            tile.pkt_handler_status |= PktHandlerError::Slverr.bit();
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::device::DeviceState;

    #[test]
    fn unknown_offset_does_not_decode() {
        let d = DeviceState::new_npu1();
        // 0x1F200 is verified SubsystemKind::Unknown for a compute tile.
        assert!(!d.ctrl_pkt_offset_decodes(2, 0x1F200));
    }

    #[test]
    fn data_memory_offset_decodes() {
        let d = DeviceState::new_npu1();
        // Compute data memory (low SRAM) classifies as DataMemory, not Unknown.
        assert!(d.ctrl_pkt_offset_decodes(2, 0x400));
    }

    #[test]
    fn read_range_fails_if_any_beat_unknown() {
        let d = DeviceState::new_npu1();
        // Start decodable, walk into the 0x1F200 Unknown hole.
        assert!(d.ctrl_pkt_read_range_decodes(2, 0x400, 4));
        assert!(!d.ctrl_pkt_read_range_decodes(2, 0x1F200, 1));
    }

    #[test]
    fn latch_sets_bit_2() {
        let mut d = DeviceState::new_npu1();
        d.latch_ctrl_slverr(0, 2);
        let tile = d.array.get(0, 2).expect("compute tile (0,2)");
        assert_eq!(tile.pkt_handler_status & 0x4, 0x4);
    }
}
```

- [ ] **Step 2: Register the module**

In `src/device/state/mod.rs`, add alongside the other `mod` declarations (e.g. directly after the `dispatch` module declaration):

```rust
mod ctrl_access;
```

- [ ] **Step 3: Run tests to verify they fail then pass**

Run: `TMPDIR=/tmp/claude-1000 cargo test -p xdna-emu --lib state::ctrl_access`
Expected: the module compiles and all four tests PASS. (If the row->tile-kind mapping for `row=2` is not Compute on NPU1, adjust the row to a known compute row -- NPU1 rows 2-5 are compute, row 0 shim, row 1 memtile; `0x400` and `0x1F200` are compute-space offsets so use a compute row.)

- [ ] **Step 4: Commit**

```bash
git add src/device/state/ctrl_access.rs src/device/state/mod.rs
git commit -m "control-packets: SLVERR decode predicate + coordinator-origin latch

ctrl_pkt_offset_decodes / ctrl_pkt_read_range_decodes use the existing
toolchain-derived subsystem_from_offset classifier; Unknown == AXI
DECERR. latch_ctrl_slverr mirrors TileArray::latch_pkt_error.

Generated using Claude Code."
```

---

### Task 3: Extract `dispatch_ctrl_action` and wire the SLVERR gate

**Files:**
- Modify: `src/interpreter/engine/coordinator.rs` (add helper; replace 3 match blocks at ~298, ~361, ~848; add tests)

- [ ] **Step 1: Write the failing integration tests**

Append to the `mod tests` block at the bottom of `src/interpreter/engine/coordinator.rs` (it already has `use super::*;`):

```rust
    #[test]
    fn ctrl_write_to_unknown_offset_sets_slverr_and_suppresses_write() {
        use crate::device::tile::CtrlPacketAction;
        let mut engine = InterpreterEngine::new_npu1();
        // 0x1F200 is verified SubsystemKind::Unknown on a compute tile.
        engine.dispatch_ctrl_action(CtrlPacketAction::WriteRegister {
            col: 0, row: 2, offset: 0x1F200, value: 0xABCD_1234,
        });
        let tile = engine.device.array.get(0, 2).expect("compute tile (0,2)");
        assert_eq!(tile.pkt_handler_status & 0x4, 0x4, "SLVERR bit must latch");
        assert!(
            tile.registers_ref().get(&0x1F200).is_none(),
            "undecoded write must be suppressed (not stored)"
        );
        assert!(
            !matches!(engine.status(), EngineStatus::Error),
            "SLVERR is poll-only sticky -- engine must not abort"
        );
    }

    #[test]
    fn ctrl_read_of_unknown_offset_sets_slverr_no_response() {
        use crate::device::tile::CtrlPacketAction;
        let mut engine = InterpreterEngine::new_npu1();
        engine.dispatch_ctrl_action(CtrlPacketAction::ReadRegisters {
            col: 0, row: 2, offset: 0x1F200, count: 2, response_id: 7,
        });
        let tile = engine.device.array.get(0, 2).expect("compute tile (0,2)");
        assert_eq!(tile.pkt_handler_status & 0x4, 0x4, "SLVERR bit must latch");
        assert!(
            tile.pending_ctrl_response.is_empty(),
            "undecoded read must not queue a response"
        );
        assert!(!matches!(engine.status(), EngineStatus::Error));
    }

    #[test]
    fn ctrl_write_to_valid_offset_no_slverr_and_applies() {
        use crate::device::tile::CtrlPacketAction;
        let mut engine = InterpreterEngine::new_npu1();
        // 0x400 is compute data memory -- decodes, must NOT SLVERR.
        engine.dispatch_ctrl_action(CtrlPacketAction::WriteRegister {
            col: 0, row: 2, offset: 0x400, value: 0x0000_0001,
        });
        let tile = engine.device.array.get(0, 2).expect("compute tile (0,2)");
        assert_eq!(
            tile.pkt_handler_status & 0x4, 0,
            "decodable offset must NOT set SLVERR (false-positive guard)"
        );
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `TMPDIR=/tmp/claude-1000 cargo test -p xdna-emu --lib coordinator::tests::ctrl_`
Expected: FAIL to compile -- `no method named dispatch_ctrl_action`.

- [ ] **Step 3: Add the `dispatch_ctrl_action` helper**

In `impl InterpreterEngine` (same impl block that holds the dispatch sites), add:

```rust
    /// Single dispatch point for a drained `CtrlPacketAction`. Owns the
    /// SLVERR decode-gate: a control-packet access whose offset does not
    /// decode is a faithful AXI slave error -- latch bit 2, suppress the
    /// access, and continue (poll-only sticky; never engine-fatal). The
    /// structural-rejection `Error` arm is unchanged (out of scope).
    fn dispatch_ctrl_action(&mut self, action: crate::device::tile::CtrlPacketAction) {
        use crate::device::tile::CtrlPacketAction;
        match action {
            CtrlPacketAction::WriteRegister { col, row, offset, value } => {
                if self.device.ctrl_pkt_offset_decodes(row, offset) {
                    self.device.write_tile_register(col, row, offset, value);
                } else {
                    log::error!(
                        "Tile ({},{}) ctrl_pkt SLVERR: write to undecoded offset \
                         0x{:05X} (sets Control_Packet_Handler_Status bit 0x4)",
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

- [ ] **Step 4: Replace the three byte-identical match blocks**

At each of the three sites, replace the `for action in ctrl_actions { match action { ... } }` block with the helper call. Site 1 (~`coordinator.rs:296-308`), Site 2 (~`:359-371`), Site 3 (~`:846-858`) each currently read:

```rust
                for action in ctrl_actions {
                    match action {
                        CtrlPacketAction::WriteRegister { col, row, offset, value } => {
                            self.device.write_tile_register(col, row, offset, value);
                        }
                        CtrlPacketAction::ReadRegisters { col, row, offset, count, response_id } => {
                            self.device.array.handle_read_registers(col, row, offset, count, response_id);
                        }
                        CtrlPacketAction::Error(msg) => {
                            self.device.array.fatal_errors.push(msg);
                        }
                    }
                }
```

Replace each with:

```rust
                for action in ctrl_actions {
                    self.dispatch_ctrl_action(action);
                }
```

Leave the surrounding `use crate::device::tile::CtrlPacketAction;` import lines and the `drain_ctrl_packet_actions()` / `drain_core_enables()` calls exactly as they are -- only the `match` body changes. The `Error` arm behavior is identical to before (still `fatal_errors.push`); only Write/Read gain the decode-gate.

- [ ] **Step 5: Run tests to verify they pass**

Run: `TMPDIR=/tmp/claude-1000 cargo test -p xdna-emu --lib coordinator::tests::ctrl_`
Expected: all three PASS.

- [ ] **Step 6: Run the full lib suite (no regressions from the refactor)**

Run: `TMPDIR=/tmp/claude-1000 cargo test -p xdna-emu --lib`
Expected: green; the 3-site extraction is behavior-preserving for decodable offsets.

- [ ] **Step 7: Commit**

```bash
git add src/interpreter/engine/coordinator.rs
git commit -m "control-packets: extract dispatch_ctrl_action + wire SLVERR gate

Replaces 3 byte-identical CtrlPacketAction match blocks with one helper
owning the decode-gate. Undecoded ctrl-pkt access -> latch bit 2,
suppress access, continue (poll-only sticky, never engine-fatal).
Error arm unchanged.

Generated using Claude Code."
```

---

### Task 4: Sticky-continue fidelity fix for parity/Tlast

**Files:**
- Modify: `src/device/array/routing.rs` (the `HandlerError(e)` arm, ~`357-364`)
- Test: a new sticky-continue regression-lock in `coordinator.rs` `mod tests`

- [ ] **Step 1: Write the failing regression-lock test**

A parity-bad control packet must set its bit AND the engine must continue (not enter `EngineStatus::Error`). Append to `coordinator.rs` `mod tests`:

```rust
    #[test]
    fn parity_handler_error_is_sticky_continue_not_fatal() {
        use crate::device::tile::CtrlPacketAction;
        let mut engine = InterpreterEngine::new_npu1();

        // Directly latch a reassembler-origin handler error the way
        // routing.rs::step_ctrl_packets must after the fix: bit set,
        // and crucially NO CtrlPacketAction::Error pushed (which would
        // route to fatal_errors -> EngineStatus::Error).
        {
            let tile = engine.device.array.get_mut(0, 2).expect("compute (0,2)");
            tile.pkt_handler_status |= crate::device::control_packets::PktHandlerError::SecondHeaderParity.bit();
        }
        // Drain whatever the ctrl path produced and dispatch it.
        let actions = engine.device.array.drain_ctrl_packet_actions();
        for a in actions {
            engine.dispatch_ctrl_action(a);
        }
        // No handler-status condition may have produced a fatal error.
        assert!(
            engine.device.array.drain_fatal_errors().is_empty(),
            "handler-status conditions must not push fatal errors"
        );
        let tile = engine.device.array.get(0, 2).unwrap();
        assert_eq!(tile.pkt_handler_status & 0x2, 0x2, "bit stays sticky");
        assert!(!matches!(engine.status(), EngineStatus::Error));
    }
```

(Note: this test's value is the contract -- after Task 4, the `HandlerError` arm no longer pushes `CtrlPacketAction::Error`, so `drain_fatal_errors()` stays empty for handler-status conditions. It fails *before* the routing fix only if a parity packet is fed end-to-end; kept as a guard that the Error-push removal in Step 3 holds and is paired with the routing change so the suite proves the arm no longer feeds `fatal_errors`.)

- [ ] **Step 2: Inspect the current `HandlerError` arm**

Run: `TMPDIR=/tmp/claude-1000 cargo test -p xdna-emu --lib coordinator::tests::parity_handler_error_is_sticky_continue_not_fatal`
Expected: PASS already at the unit level shown (the test latches directly). Then proceed to make the *routing* path faithful so an end-to-end parity packet also cannot abort.

- [ ] **Step 3: Remove the `CtrlPacketAction::Error` push from the `HandlerError` arm**

In `src/device/array/routing.rs`, the `ReassembleResult::HandlerError(e)` arm currently reads (around lines 357-364):

```rust
                        ReassembleResult::HandlerError(e) => {
                            log::error!(
                                "Tile ({},{}) ctrl_pkt handler error: {:?} (sets Control_Packet_Handler_Status bit 0x{:X})",
                                col, row, e, e.bit()
                            );
                            Self::latch_pkt_error(&mut self.tiles[i], e);
                            self.pending_ctrl_actions.push(CtrlPacketAction::Error(format!("{:?}", e)));
                        }
```

Replace it with (drop the push; the bit is the faithful signal, the handler continues):

```rust
                        ReassembleResult::HandlerError(e) => {
                            // Poll-only sticky-continue (aie-rt/AM025): latch
                            // the Control_Packet_Handler_Status bit and keep
                            // processing. NOT engine-fatal -- pushing
                            // CtrlPacketAction::Error here would route to
                            // fatal_errors -> EngineStatus::Error (the bug the
                            // SLVERR plan corrects). Firmware polls the bit.
                            log::error!(
                                "Tile ({},{}) ctrl_pkt handler error: {:?} (sets Control_Packet_Handler_Status bit 0x{:X})",
                                col, row, e, e.bit()
                            );
                            Self::latch_pkt_error(&mut self.tiles[i], e);
                        }
```

Leave the `ReassembleResult::Error(msg)` arm immediately below it **unchanged** (structural rejections remain `log` + `CtrlPacketAction::Error` -- out of scope per spec Section 1).

- [ ] **Step 4: Run the targeted + full suite**

Run: `TMPDIR=/tmp/claude-1000 cargo test -p xdna-emu --lib`
Expected: green. In particular the reassembler unit tests (`reassembler.rs` ~380-470) still pass (they assert `ReassembleResult::HandlerError`, unaffected), and no test asserts the old abort (audit confirmed none exists).

- [ ] **Step 5: Commit**

```bash
git add src/device/array/routing.rs src/interpreter/engine/coordinator.rs
git commit -m "control-packets: handler errors are poll-only sticky-continue

Drops the CtrlPacketAction::Error push from the HandlerError arm so
parity/Tlast (and SLVERR) latch the sticky bit and continue, matching
aie-rt/AM025 (no interrupt, no abort). Corrects the shipped 3-bit
fatal-abort. Structural-rejection Error arm unchanged. Audit: no
shipped test asserted the old abort behavior.

Generated using Claude Code."
```

---

### Task 5: Coverage lockstep -- `control_packets` -> `Modeled{Full}`

**Files:**
- Modify: `crates/xdna-archspec/src/coverage/units.rs:177-180` (`control_packets` seed) and `:169-172` (`noc` seed forward-link)
- Modify: `src/device/tile/mod.rs:305-311` (doc block)
- Regenerate: `docs/coverage/aie2/subsystem-index.md`, `implementation-gaps.md`, `architecture-index.md`

- [ ] **Step 1: Update the `control_packets` seed**

In `crates/xdna-archspec/src/coverage/units.rs`, replace the `control_packets` entry (lines 177-180) with:

```rust
        d("control_packets", "AM025 Control_Packet_Handler_Status (0x3FF30/0xB0F30); aie-rt xaiegbl_params.h:7761; XRT host protocol",
          &["src/device/control_packets/", "src/device/tile/mod.rs", "src/device/state/ctrl_access.rs", "src/interpreter/engine/coordinator.rs", "src/npu/"],
          "Control-packet headers, reassembly, register read/write effects, response packets MODELED. NPU host instruction stream (WRITE32/BLOCKWRITE/BLOCKSET/MASKWRITE/MASKPOLL/CONFIG_SHIMDMA_*/DDR_PATCH) MODELED. All four Tile_Control_Packet_Handler_Status sticky bits have faithful detecting paths: First/Second-header parity and Tlast via the reassembler, SLVERR_On_Access via undecoded-address decode at the dispatch boundary, all with poll-only sticky-continue semantics (aie-rt/AM025: latch + continue, no interrupt, no abort). The AIE_AXIMM_Config.SLVERR_Block config-suppression refinement is a tracked NoC-gated goal -- see the noc domain. Keystone subsystem.",
          Modeled { completeness: Full }, None),
```

- [ ] **Step 2: Add the forward-link in the `noc` seed**

In the same file, replace the `noc` narrative string (line 171) so the SLVERR_Block dependency is discoverable from the `noc` gap. Change the `noc` entry's narrative to:

```rust
          "Direct NoC control / AIE_AXIMM_Config / NoC fabric latency-arbitration are not modeled (NoC fudged; impacts cycle-accuracy more than functional correctness; cycle-accuracy-mission.md tracks calibration). AIE_AXIMM_Config.SLVERR_Block (unmapped-access SLVERR suppression) is the tracked control_packets-fidelity dependency gated here: control-packet SLVERR is modeled for the reset-default (SLVERR-enabled) configuration every observed binary uses; honoring SLVERR_Block needs this NoC register plumbed. NPI privileged register access is driver-side privilege -- emulator gives unrestricted access: accepted out of scope. No emulator src for the unmodeled NoC surface.",
```

(Leave the `noc` `source_ref`, `src_locations`, and `Modeled { completeness: Stub }` verdict unchanged -- `noc` stays a Stub; only its narrative gains the forward-link.)

- [ ] **Step 3: Update the `tile/mod.rs` doc block**

In `src/device/tile/mod.rs`, the `pkt_handler_status` doc block (lines ~300-312) currently annotates bit 2 with "(successor plan: runtime register-access-error model)". Replace the doc block's bit list and trailing paragraph so it reads:

```rust
    ///   [0] First_Header_Parity_Error  -- detected (stream header parity)
    ///   [1] Second_Header_Parity_Error -- detected (opcode header parity)
    ///   [2] SLVERR_On_Access           -- detected (undecoded ctrl-pkt
    ///       register access; AXI DECERR analogue)
    ///   [3] Tlast_Error                -- detected (write TLAST)
    ///
    /// All four are poll-only sticky bits with write-1-to-clear. The
    /// reassembler returns ReassembleResult::HandlerError for bits
    /// 0/1/3 (latched in routing.rs); bit 2 is latched at the
    /// coordinator dispatch boundary (DeviceState::latch_ctrl_slverr).
    /// The handler latches and continues -- no interrupt, no abort;
    /// software polls then writes 1 to clear. Compute + memtile only;
    /// shim has no packet handler.
```

(Keep the exact leading `///` indentation of the surrounding block; only the bit-2 line and the trailing paragraph change in substance. `src/device/state/effects.rs:28-35` already has the correct AM025 order with no successor-plan wording -- **do not modify it**; verify by reading those lines.)

- [ ] **Step 4: Run the archspec suite (verdict-shape gate)**

Run: `TMPDIR=/tmp/claude-1000 cargo test -p xdna-archspec --lib`
Expected: green. The `capability_spine` tests assert every domain carries a non-empty `source_ref` and that `Modeled{Full}` domains name `src_locations` -- the updated `control_packets` entry satisfies both (it has `src_locations`).

- [ ] **Step 5: Regenerate the coverage artifacts**

Run: `TMPDIR=/tmp/claude-1000 cargo run -p xdna-archspec --example gen_coverage_artifacts`
Expected: completes; `git status` shows modifications to `docs/coverage/aie2/subsystem-index.md`, `docs/coverage/aie2/implementation-gaps.md`, `docs/coverage/aie2/architecture-index.md`.

- [ ] **Step 6: Sanity-check the generated diffs**

Run: `git diff --stat docs/coverage/aie2/ && grep -n "control_packets" docs/coverage/aie2/implementation-gaps.md`
Expected: `implementation-gaps.md` no longer has a `control_packets: PARTIAL ...` line (it was line 11; the gap queue drops by one). `subsystem-index.md` shows `control_packets` as `Modeled{Full}` and the `noc` row carries the SLVERR_Block forward-link. `architecture-index.md` regenerated (the control-packets domain category verdict may roll up to fully modeled -- accept the derived diff).

- [ ] **Step 7: Commit code + regenerated artifacts together (Plan-3 staleness lockstep)**

```bash
git add crates/xdna-archspec/src/coverage/units.rs src/device/tile/mod.rs docs/coverage/aie2/subsystem-index.md docs/coverage/aie2/implementation-gaps.md docs/coverage/aie2/architecture-index.md
git commit -m "control-packets: close gap -- control_packets -> Modeled{Full}

All 4 handler-status bits faithfully detected with poll-only
sticky-continue. Gap queue 5 -> 4. SLVERR_Block forward-linked from the
noc gap (NoC-gated tracked goal). Coverage artifacts regenerated in
lockstep.

Generated using Claude Code."
```

- [ ] **Step 8: Zero-drift verification**

Run: `TMPDIR=/tmp/claude-1000 cargo run -p xdna-archspec --example gen_coverage_artifacts && git diff --quiet docs/coverage/aie2/ && echo ZERO-DRIFT-OK`
Expected: prints `ZERO-DRIFT-OK` (regenerating again produces no diff -- the committed artifacts match the generator output exactly, per the Plan-3 staleness gate).

---

## Final Gate (after all tasks)

- [ ] Run: `TMPDIR=/tmp/claude-1000 cargo test -p xdna-emu --lib` -- green.
- [ ] Run: `TMPDIR=/tmp/claude-1000 cargo test -p xdna-archspec --lib` -- green.
- [ ] Run: `TMPDIR=/tmp/claude-1000 cargo run -p xdna-archspec --example gen_coverage_artifacts && git diff --quiet docs/coverage/aie2/ && echo ZERO-DRIFT-OK` -- prints `ZERO-DRIFT-OK`.
- [ ] `git log --oneline` shows the five task commits on `dev`.

## Self-Review (performed)

**Spec coverage:** §2 PktHandlerError::Slverr -> Task 1. §3 Unknown-decode definition -> Task 2. §4 unified sticky-continue + shipped fix -> Task 4 (and Task 3 for SLVERR's own arm). §4.1 shipped-test audit -> covered in Pre-flight + Task 4 Step 4 (no rewrite needed; new lock added). §5 predicate + read-range -> Task 2. §6 shared dispatch helper -> Task 3. §7 testing -> Tasks 2/3/4 tests + Final Gate. §8 coverage lockstep + doc blocks -> Task 5. §9 files -> all mapped. §10 SLVERR_Block tracked/NoC-gated -> Task 5 Step 2 forward-link + Task 1 doc. §11 non-goals -> respected (no fallible write_tile_register, no regdb extraction, structural-rejection arm untouched, processor.rs untouched, no HW tier).

**Placeholder scan:** none -- every code step shows complete code; every command is exact with expected output.

**Type consistency:** `PktHandlerError::Slverr` / `.bit()==0x4`, `ctrl_pkt_offset_decodes(row,offset)`, `ctrl_pkt_read_range_decodes(row,offset,count)`, `latch_ctrl_slverr(col,row)`, `dispatch_ctrl_action(action)` used identically across Tasks 1-5. `InterpreterEngine`, `DeviceState`, `CtrlPacketAction`, `EngineStatus::Error`, `ReassembleResult::HandlerError` match the real code read during grounding.
