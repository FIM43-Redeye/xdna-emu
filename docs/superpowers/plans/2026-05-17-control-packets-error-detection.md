# Control-Packet Handler Error Detection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire faithful detecting paths + sticky bits for First_Header_Parity (0), Second_Header_Parity tighten (1), and Tlast_Error (3) in `Tile_Control_Packet_Handler_Status`, correct the `ID_Parity` misnomer, and narrow the `control_packets` coverage gap to SLVERR-only.

**Architecture:** A 3-variant `PktHandlerError` enum is the single source of truth for the bit map. The reassembler detects the three protocol violations and returns a new `ReassembleResult::HandlerError(PktHandlerError)`; `routing.rs` has one latch helper that ORs `err.bit()` into the destination tile's `pkt_handler_status`. Odd-parity math is DRY-extracted into `odd_parity_ok` in `packet_types.rs`. SLVERR (bit 2) is out of scope -- the runtime register-access path cannot fail today; it is the committed successor plan.

**Tech Stack:** Rust, `cargo test --lib`, xdna-archspec coverage generator.

**Spec:** `docs/superpowers/specs/2026-05-17-control-packets-error-detection-design.md`

**Branch:** `dev` (durable; this is incremental work on it, consistent with prior coverage/feature work).

**Sandbox note:** run tests as `TMPDIR=/tmp/claude-1000 cargo test --lib` so temp-dir tests pass in the sandbox.

---

## Pre-flight context for the implementer

You have zero prior context. Key facts about this codebase:

- The control-packet runtime path is: `routing.rs::step_ctrl_packets` -> `StreamReassembler::feed_word` -> on `Complete` -> `packet_to_actions()` -> `CtrlPacketAction` -> `coordinator.rs`. `ControlPacketProcessor::process()` is **dead code at runtime** (test-only) -- do not touch it.
- `StreamReassembler` (`src/device/control_packets/reassembler.rs`) is a 3-state machine: `WaitingForStreamHeader` (only when `drop_header=false`) -> `Idle` (parses control-packet opcode header) -> `Collecting` (gathers data beats). `feed_word(word, tlast)` returns `ReassembleResult`.
- Odd parity in this codebase is always `word.count_ones() & 1 == 1` (appears inline at `packet_types.rs` `decode:165` and `encode:140`). The stream routing header is parsed by `PacketHeader::decode(word) -> (PacketHeader, parity_ok)` in `src/device/stream_switch/packet_types.rs`.
- `pkt_handler_status: u32` is a field on `Tile` (`src/device/tile/mod.rs`). Write-1-to-clear is already generic in `src/device/state/effects.rs` (`tile.pkt_handler_status &= !(value & 0xF)`); the register read path is already generic in `src/device/tile/registers.rs` (`return self.pkt_handler_status & 0xF`). Do not change either's behavior.
- Project conventions: no `_` wildcard arms on the new enum's matches (a missing variant must be a compile error); comments explain *why*; no emoji; commit messages end with `Generated using Claude Code.`; bare `cargo` is ground truth (rust-analyzer harness diagnostics can be stale -- verify with a real build/test, not the diagnostic panel).

---

## Task 1: DRY-extract `odd_parity_ok`

**Files:**
- Modify: `src/device/stream_switch/packet_types.rs` (add fn; refactor `PacketHeader::decode` ~line 165)
- Test: same file, `#[cfg(test)] mod tests`

- [ ] **Step 1: Write the failing test**

Add to the tests module in `src/device/stream_switch/packet_types.rs`:

```rust
#[test]
fn odd_parity_ok_basic() {
    // count_ones() odd  -> true ; even -> false
    assert!(odd_parity_ok(0b1));          // 1 one  -> odd
    assert!(!odd_parity_ok(0b11));        // 2 ones -> even
    assert!(odd_parity_ok(0b111));        // 3 ones -> odd
    assert!(!odd_parity_ok(0));           // 0 ones -> even
    assert!(odd_parity_ok(0xFFFF_FFFF) == (32 % 2 == 1)); // 32 ones -> even -> false
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib odd_parity_ok_basic`
Expected: FAIL to compile -- `cannot find function odd_parity_ok`.

- [ ] **Step 3: Add the helper and refactor `decode`**

Add near the top of the `impl PacketHeader` block's module (free function, above `impl PacketHeader`):

```rust
/// Odd-parity check over a full 32-bit header word.
///
/// The AIE packet/control-packet headers use odd parity: the total
/// number of set bits (including the parity bit) is odd on a valid
/// header. Single source of truth for the parity formula -- both
/// `PacketHeader::decode` and the control-packet reassembler call this.
pub fn odd_parity_ok(word: u32) -> bool {
    word.count_ones() & 1 == 1
}
```

In `PacketHeader::decode`, replace the inline parity line (currently
`let parity_ok = word.count_ones() & 1 == 1;`) with:

```rust
        let parity_ok = odd_parity_ok(word);
```

- [ ] **Step 4: Run tests to verify pass + no behavior change**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib -p xdna-emu packet_types`
Expected: PASS, including all pre-existing `decode`/`encode` tests (the refactor is behavior-preserving).

- [ ] **Step 5: Commit**

```bash
git add src/device/stream_switch/packet_types.rs
git commit -m "control-packets: DRY-extract odd_parity_ok parity helper

Single source of truth for the count_ones() & 1 == 1 odd-parity
formula; PacketHeader::decode now calls it. Behavior-preserving,
locked by existing decode/encode tests.

Generated using Claude Code."
```

---

## Task 2: `PktHandlerError` enum

**Files:**
- Create: `src/device/control_packets/status.rs`
- Modify: `src/device/control_packets/mod.rs` (register module + re-export, near lines 57-65)
- Test: `src/device/control_packets/status.rs` `#[cfg(test)] mod tests`

- [ ] **Step 1: Write the failing test**

Create `src/device/control_packets/status.rs` containing only:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_map_matches_am025() {
        assert_eq!(PktHandlerError::FirstHeaderParity.bit(), 0x1);
        assert_eq!(PktHandlerError::SecondHeaderParity.bit(), 0x2);
        assert_eq!(PktHandlerError::Tlast.bit(), 0x8);
    }
}
```

Add to `src/device/control_packets/mod.rs` after the other `pub mod`
lines (after `pub mod response;`):

```rust
pub mod status;
```

and after the other `pub use` lines (after the `pub use response::...`):

```rust
pub use status::PktHandlerError;
```

- [ ] **Step 2: Run it to verify it fails**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib bit_map_matches_am025`
Expected: FAIL to compile -- `PktHandlerError` not found.

- [ ] **Step 3: Implement the enum**

Prepend to `src/device/control_packets/status.rs` (above the test
module):

```rust
//! Tile_Control_Packet_Handler_Status sticky-error conditions.

/// A control-packet handler error with a faithful detecting path.
///
/// Bit positions per AM025 `Tile_Control_Packet_Handler_Status`
/// (regdb fields `First_Header_Parity_Error` /
/// `Second_Header_Parity_Error` / `Tlast_Error`). This is the single
/// source of truth for the bit map -- no other site names these
/// positions.
///
/// `SLVERR_On_Access` (bit `0x4`) is intentionally absent: it has no
/// faithful trigger until the successor plan repairs the runtime
/// register-access path. That plan adds the `Slverr` variant here; the
/// exhaustive no-`_` match below will then force every consumer to
/// handle it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PktHandlerError {
    /// Stream routing header (First) odd-parity failure -- bit 0.
    FirstHeaderParity,
    /// Control-packet opcode header (Second) odd-parity failure -- bit 1.
    SecondHeaderParity,
    /// TLAST in the wrong position on a write-class packet -- bit 3.
    Tlast,
}

impl PktHandlerError {
    /// The `Tile_Control_Packet_Handler_Status` bit this condition sets.
    pub fn bit(self) -> u32 {
        match self {
            PktHandlerError::FirstHeaderParity => 0x1,
            PktHandlerError::SecondHeaderParity => 0x2,
            PktHandlerError::Tlast => 0x8,
        }
    }
}
```

- [ ] **Step 4: Run tests to verify pass**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib bit_map_matches_am025`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/device/control_packets/status.rs src/device/control_packets/mod.rs
git commit -m "control-packets: add PktHandlerError enum (3 faithful bits)

Single source of truth for the Tile_Control_Packet_Handler_Status bit
map. SLVERR deliberately absent (successor plan); exhaustive no-_ match
makes adding it a forced compile error at every consumer.

Generated using Claude Code."
```

---

## Task 3: `HandlerError` channel + routing latch rewrite (bit-1 tightening)

This adds the error channel and rewrites the single latch point. After
this task, structural `ReassembleResult::Error` no longer sets any
status bit (the bit-1 tightening); no detector produces `HandlerError`
yet, so `cargo test --lib` must stay green (the `Error` arm is
practically unreachable from `feed_word` -- `parse_header` only errors
on a 2-bit-impossible opcode and `beats` is always >= 1 -- so removing
its bit set is observably inert until Task 5 adds parity detection).

**Files:**
- Modify: `src/device/control_packets/reassembler.rs` (`ReassembleResult` enum, ~lines 52-60)
- Modify: `src/device/array/routing.rs` (latch helper + match arm, ~lines 347-366)
- Test: `src/device/array/routing.rs` `#[cfg(test)] mod tests` (add if absent) or an inline unit test in `reassembler.rs`

- [ ] **Step 1: Write the failing test**

Add to the tests module in `src/device/control_packets/reassembler.rs`:

```rust
#[test]
fn handler_error_variant_carries_pkt_handler_error() {
    use crate::device::control_packets::status::PktHandlerError;
    let r = ReassembleResult::HandlerError(PktHandlerError::SecondHeaderParity);
    match r {
        ReassembleResult::HandlerError(e) => assert_eq!(e.bit(), 0x2),
        other => panic!("expected HandlerError, got {:?}", other),
    }
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib handler_error_variant_carries_pkt_handler_error`
Expected: FAIL to compile -- no `HandlerError` variant.

- [ ] **Step 3: Add the variant**

In `src/device/control_packets/reassembler.rs`, extend the
`ReassembleResult` enum (currently `Pending` / `Complete` / `Error`):

```rust
/// Result of feeding a word to the reassembler.
#[derive(Debug)]
pub enum ReassembleResult {
    /// Still collecting -- no complete packet yet.
    Pending,
    /// A complete control packet is ready.
    Complete(ControlPacket),
    /// A protocol violation with a faithful Control_Packet_Handler_Status
    /// detecting path (parity / TLAST). Latched as a sticky bit.
    HandlerError(super::status::PktHandlerError),
    /// A structural rejection (logged only, no status bit).
    Error(String),
}
```

- [ ] **Step 4: Rewrite the routing latch**

In `src/device/array/routing.rs`, add an associated helper to the same
`impl` block that contains `step_ctrl_packets` (place it next to
`packet_to_actions`):

```rust
    /// The single Control_Packet_Handler_Status latch point. All faithful
    /// handler errors converge here so the bit map lives in exactly one
    /// place (`PktHandlerError::bit()`).
    fn latch_pkt_error(tile: &mut crate::device::tile::Tile, e: crate::device::control_packets::status::PktHandlerError) {
        tile.pkt_handler_status |= e.bit();
    }
```

Replace the current match arm (currently:
```rust
                        ReassembleResult::Error(msg) => {
                            log::error!("{}", msg);
                            // Set Second_Header_Parity_Error sticky bit so
                            // software polling Control_Packet_Handler_Status
                            // sees a non-zero value. Bit 1 covers header
                            // parse failures specifically.
                            self.tiles[i].pkt_handler_status |= 0x2;
                            self.pending_ctrl_actions.push(CtrlPacketAction::Error(msg));
                        }
```
) with:

```rust
                        ReassembleResult::HandlerError(e) => {
                            log::error!(
                                "Tile ({},{}) ctrl_pkt handler error: {:?} (sets Control_Packet_Handler_Status bit 0x{:X})",
                                col, row, e, e.bit()
                            );
                            Self::latch_pkt_error(&mut self.tiles[i], e);
                            self.pending_ctrl_actions
                                .push(CtrlPacketAction::Error(format!("{:?}", e)));
                        }
                        ReassembleResult::Error(msg) => {
                            // Structural rejection: logged only. Per spec
                            // 3.2 it does NOT set Second_Header_Parity --
                            // that bit is now exclusively a true-parity
                            // signal (Task 5). No pkt_handler_status write.
                            log::error!("{}", msg);
                            self.pending_ctrl_actions.push(CtrlPacketAction::Error(msg));
                        }
```

- [ ] **Step 5: Run tests to verify pass**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib`
Expected: PASS for xdna-emu (the new test passes; existing tests
unaffected -- the removed `|= 0x2` was on a practically unreachable
arm).

- [ ] **Step 6: Commit**

```bash
git add src/device/control_packets/reassembler.rs src/device/array/routing.rs
git commit -m "control-packets: HandlerError channel + single latch point

Adds ReassembleResult::HandlerError(PktHandlerError); routing.rs gets
one latch_pkt_error helper. Structural Error no longer sets bit 1 --
the bit-1 tightening (bit 1 becomes exclusively a true-parity signal
once Task 5 lands).

Generated using Claude Code."
```

---

## Task 4: First_Header_Parity detection (bit 0)

**Files:**
- Modify: `src/device/control_packets/reassembler.rs` (`WaitingForStreamHeader` arm of `feed_word`, ~lines 98-112)
- Test: same file, tests module

- [ ] **Step 1: Write the failing test**

Add to the reassembler tests module:

```rust
#[test]
fn bad_parity_stream_header_sets_first_header_parity() {
    use crate::device::control_packets::status::PktHandlerError;
    let mut r = StreamReassembler::new(0, 2);
    r.set_drop_header(false); // stream header expected

    // Stream header with EVEN ones -> odd-parity invalid.
    // 0b11 has two set bits (even) -> odd_parity_ok == false.
    match r.feed_word(0b11, false) {
        ReassembleResult::HandlerError(PktHandlerError::FirstHeaderParity) => {}
        other => panic!("expected FirstHeaderParity, got {:?}", other),
    }
}

#[test]
fn good_parity_stream_header_is_consumed() {
    let mut r = StreamReassembler::new(0, 2);
    r.set_drop_header(false);
    // 0b1 has one set bit (odd) -> odd_parity_ok == true -> consumed.
    assert!(matches!(r.feed_word(0b1, false), ReassembleResult::Pending));
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib bad_parity_stream_header good_parity_stream_header`
Expected: FAIL -- `bad_parity_stream_header_sets_first_header_parity`
gets `Pending` (no parity check yet).

- [ ] **Step 3: Implement detection**

In `src/device/control_packets/reassembler.rs`, add imports at the top
(near `use super::parser::...`):

```rust
use super::status::PktHandlerError;
use crate::device::stream_switch::packet_types::{odd_parity_ok, PacketHeader};
```

Replace the `ReassemblerState::WaitingForStreamHeader` arm body with:

```rust
            ReassemblerState::WaitingForStreamHeader => {
                // The stream routing header is exactly what
                // PacketHeader::decode parses. Validate its odd parity
                // (First_Header_Parity). Reachable only when
                // drop_header=false; when the switch dropped the header
                // the handler never sees it and this cannot fire.
                let (hdr, parity_ok) = PacketHeader::decode(word);
                if !parity_ok {
                    // Stay waiting for a valid header.
                    self.state = ReassemblerState::WaitingForStreamHeader;
                    return ReassembleResult::HandlerError(PktHandlerError::FirstHeaderParity);
                }
                log::debug!(
                    "Tile ({},{}) ctrl_pkt: consuming stream header 0x{:08X} (stream_id={}, type={:?})",
                    self.col, self.row, word, hdr.stream_id, hdr.packet_type
                );
                self.state = ReassemblerState::Idle;
                ReassembleResult::Pending
            }
```

(If the `odd_parity_ok` import is unused after this task, that is fine
-- Task 5 uses it; keep the import.)

- [ ] **Step 4: Run tests to verify pass**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib -p xdna-emu reassembler`
Expected: PASS, including the pre-existing
`stream_header_consumed_when_drop_false` test (its header `0x0000_0007`
has three set bits -> odd -> still consumed).

- [ ] **Step 5: Commit**

```bash
git add src/device/control_packets/reassembler.rs
git commit -m "control-packets: detect First_Header_Parity (bit 0)

Stream routing header now validated via PacketHeader::decode parity_ok;
bad parity returns HandlerError(FirstHeaderParity). Reachable only when
drop_header=false.

Generated using Claude Code."
```

---

## Task 5: Second_Header_Parity detection + tightening (bit 1)

**Files:**
- Modify: `src/device/control_packets/reassembler.rs` (`Idle` arm of `feed_word`, ~lines 113-156)
- Test: same file, tests module

- [ ] **Step 1: Write the failing test**

Add to the reassembler tests module:

```rust
#[test]
fn bad_parity_ctrl_header_sets_second_header_parity() {
    use crate::device::control_packets::status::PktHandlerError;
    let mut r = StreamReassembler::new(0, 2); // drop_header=true: starts Idle

    // build_test_header sets no parity bit. Pick fields so the assembled
    // word has an EVEN number of set bits -> odd-parity invalid.
    // addr=0x100 (1 one), len=0, op=0, resp=0 -> total ones = 1 (odd, VALID).
    // Add one more set bit via address to make it even: addr=0x101 -> 2 ones.
    let header = build_test_header(0x101, 0, 0, 0); // 0x101 -> 2 set bits -> even -> invalid
    match r.feed_word(header, false) {
        ReassembleResult::HandlerError(PktHandlerError::SecondHeaderParity) => {}
        other => panic!("expected SecondHeaderParity, got {:?}", other),
    }
}

#[test]
fn good_parity_ctrl_header_proceeds() {
    let mut r = StreamReassembler::new(0, 2);
    // addr=0x100 -> 1 set bit (odd) -> valid -> goes Collecting (Pending).
    let header = build_test_header(0x100, 0, 0, 0);
    assert!(matches!(r.feed_word(header, false), ReassembleResult::Pending));
}
```

(Note for the implementer: `build_test_header` is the existing helper at
the bottom of the tests module; it leaves the parity bit 0. Verify the
set-bit counts with `u32::count_ones` if a literal's parity is unclear
-- the assertion is what matters, adjust the address literal so the
assembled word's `count_ones()` is even for the bad case and odd for the
good case.)

- [ ] **Step 2: Run it to verify it fails**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib bad_parity_ctrl_header good_parity_ctrl_header`
Expected: FAIL -- bad-parity case currently parses and returns
`Pending` (no parity gate yet).

- [ ] **Step 3: Implement the parity gate**

In `src/device/control_packets/reassembler.rs`, at the very start of the
`ReassemblerState::Idle` arm (before `let header = match parse_header`):

```rust
            ReassemblerState::Idle => {
                // Second_Header_Parity: validate opcode-header odd parity
                // BEFORE structural decode -- hardware checks header parity
                // at ingress, before acting on opcode/length. Structural
                // rejections (below) are logged only and do NOT set this
                // bit (spec 3.2 tightening).
                if !odd_parity_ok(word) {
                    self.state = ReassemblerState::Idle;
                    return ReassembleResult::HandlerError(PktHandlerError::SecondHeaderParity);
                }

                // Parse control packet header.
                let header = match parse_header(word) {
                    Ok(h) => h,
                    Err(e) => {
                        self.state = ReassemblerState::Idle;
                        return ReassembleResult::Error(format!(
                            "Tile ({},{}) ctrl_pkt: header parse error: {}",
                            self.col, self.row, e
                        ));
                    }
                };
```

(Leave the rest of the `Idle` arm -- the `log::info!`, the OP_READ
fast-path, the zero-beats guard, the transition to `Collecting` --
exactly as is.)

- [ ] **Step 4: Run tests to verify pass**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib -p xdna-emu reassembler`
Expected: PASS. Pre-existing reassembler tests that build headers with
`build_test_header` may now hit the parity gate -- if any pre-existing
test fails because its test header has even set-bit count, fix the test
header literal (or feed a parity-correct header) so the test exercises
its intended path; do **not** weaken the parity gate. Document any such
test adjustment in the commit message.

- [ ] **Step 5: Run the full suite**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib`
Expected: PASS for xdna-emu. (Bit 1 is now exclusively a true-parity
signal; structural `Error` sets nothing -- the tightening is complete.)

- [ ] **Step 6: Commit**

```bash
git add src/device/control_packets/reassembler.rs
git commit -m "control-packets: detect Second_Header_Parity + tighten bit 1

Opcode-header odd parity validated before structural decode; bad parity
returns HandlerError(SecondHeaderParity). Bit 1 is now exclusively a
true-parity signal -- structural rejections set no status bit.

Generated using Claude Code."
```

---

## Task 6: Tlast_Error detection (bit 3)

**Files:**
- Modify: `src/device/control_packets/reassembler.rs` (`Collecting` arm of `feed_word`, ~lines 157-189)
- Test: same file, tests module

- [ ] **Step 1: Write the failing test**

Add to the reassembler tests module:

```rust
#[test]
fn missing_final_tlast_on_write_sets_tlast_error() {
    use crate::device::control_packets::status::PktHandlerError;
    let mut r = StreamReassembler::new(0, 2);
    // 1-beat write (addr 0x100 -> odd parity OK), final beat WITHOUT tlast.
    let header = build_test_header(0x100, 0, 0, 0);
    assert!(matches!(r.feed_word(header, false), ReassembleResult::Pending));
    match r.feed_word(0xDEAD_BEEF, false) {
        ReassembleResult::HandlerError(PktHandlerError::Tlast) => {}
        other => panic!("expected Tlast, got {:?}", other),
    }
}

#[test]
fn early_tlast_on_multibeat_write_sets_tlast_error() {
    use crate::device::control_packets::status::PktHandlerError;
    let mut r = StreamReassembler::new(0, 2);
    // 3-beat write (length field 2). TLAST asserted on beat 1 (early).
    let header = build_test_header(0x100, 2, 0, 0);
    assert!(matches!(r.feed_word(header, false), ReassembleResult::Pending));
    match r.feed_word(0x11, true) {
        ReassembleResult::HandlerError(PktHandlerError::Tlast) => {}
        other => panic!("expected Tlast, got {:?}", other),
    }
}

#[test]
fn correct_final_tlast_completes_normally() {
    let mut r = StreamReassembler::new(0, 2);
    let header = build_test_header(0x100, 0, 0, 0);
    r.feed_word(header, false);
    match r.feed_word(0x42, true) {
        ReassembleResult::Complete(pkt) => assert_eq!(pkt.data[0], 0x42),
        other => panic!("expected Complete, got {:?}", other),
    }
}

#[test]
fn read_packet_needs_no_tlast() {
    // OP_READ completes at the header; TLAST is irrelevant -- no Tlast error.
    let mut r = StreamReassembler::new(0, 2);
    let header = build_test_header(0x300, 0, 1, 7); // op=1 Read
    match r.feed_word(header, false) {
        ReassembleResult::Complete(pkt) => assert_eq!(pkt.opcode, CtrlOpCode::Read),
        other => panic!("expected Complete, got {:?}", other),
    }
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib missing_final_tlast early_tlast correct_final_tlast read_packet_needs_no_tlast`
Expected: FAIL -- the two error cases currently return `Complete`
(no TLAST validation).

- [ ] **Step 3: Implement TLAST validation**

In `src/device/control_packets/reassembler.rs`, in the
`ReassemblerState::Collecting` arm, after `beats_collected += 1;` and the
existing `log::debug!`, replace the completion block. The current code
is:

```rust
                if beats_collected >= header.beats {
                    // All beats received -- build the complete packet.
                    let payload = &data[..beats_collected as usize];
                    let packet = match header.opcode {
                        CtrlOpCode::Write | CtrlOpCode::BlockWrite => {
                            ControlPacket::block_write(header.address, payload.to_vec())
                        }
                        CtrlOpCode::WriteIncr => ControlPacket::write_incr(header.address, payload.to_vec()),
                        CtrlOpCode::Read => unreachable!("Read handled above"),
                    };
                    self.transition_after_complete(tlast);
                    ReassembleResult::Complete(packet)
                } else {
                    // Still collecting.
                    self.state = ReassemblerState::Collecting { header, beats_collected, data };
                    ReassembleResult::Pending
                }
```

Replace it with:

```rust
                let is_final_beat = beats_collected >= header.beats;

                // TLAST must land exactly on the final declared beat for
                // write-class packets ("missing or unexpected TLAST").
                // Read packets never reach Collecting (handled at header).
                if tlast && !is_final_beat {
                    // Unexpected early TLAST.
                    self.transition_after_complete(true);
                    return ReassembleResult::HandlerError(PktHandlerError::Tlast);
                }
                if is_final_beat && !tlast {
                    // Missing TLAST on the final beat.
                    self.transition_after_complete(false);
                    return ReassembleResult::HandlerError(PktHandlerError::Tlast);
                }

                if is_final_beat {
                    // All beats received with correct TLAST -- build packet.
                    let payload = &data[..beats_collected as usize];
                    let packet = match header.opcode {
                        CtrlOpCode::Write | CtrlOpCode::BlockWrite => {
                            ControlPacket::block_write(header.address, payload.to_vec())
                        }
                        CtrlOpCode::WriteIncr => ControlPacket::write_incr(header.address, payload.to_vec()),
                        CtrlOpCode::Read => unreachable!("Read handled above"),
                    };
                    self.transition_after_complete(tlast);
                    ReassembleResult::Complete(packet)
                } else {
                    // Still collecting.
                    self.state = ReassemblerState::Collecting { header, beats_collected, data };
                    ReassembleResult::Pending
                }
```

- [ ] **Step 4: Run tests to verify pass**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib -p xdna-emu reassembler`
Expected: PASS. Pre-existing reassembler tests feed a correct final
TLAST (e.g. `single_word_write` calls `feed_word(0xDEADBEEF, true)`), so
they remain green. If any pre-existing multi-beat test fed a
non-final-with-tlast or final-without-tlast pattern, it was relying on
unchecked TLAST -- correct the test's `tlast` flags to a valid sequence
and note it in the commit.

- [ ] **Step 5: Run the full suite**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib`
Expected: PASS for xdna-emu.

- [ ] **Step 6: Commit**

```bash
git add src/device/control_packets/reassembler.rs
git commit -m "control-packets: detect Tlast_Error (bit 3)

Write-class packets must assert TLAST exactly on the final declared
beat; early or missing TLAST returns HandlerError(Tlast). Read packets
unaffected (complete at header).

Generated using Claude Code."
```

---

## Task 7: Misnomer fixes + coverage lockstep

**Files:**
- Modify: `src/device/tile/mod.rs` (the `pkt_handler_status` doc block, ~lines 298-311)
- Modify: `src/device/state/effects.rs` (comment, ~lines 28-30)
- Modify: `crates/xdna-archspec/src/coverage/units.rs` (the `control_packets` seed, ~lines 177-180)
- Modify: `src/device/tile/registers.rs` (extend `pkt_handler_status_tests`, ~lines 345-369)
- Regenerate: `docs/coverage/aie2/{subsystem-index,implementation-gaps,architecture-index}.md`

- [ ] **Step 1: Write the failing tests (register read-back + wtc for bits 0 and 3)**

Add to `mod pkt_handler_status_tests` in `src/device/tile/registers.rs`:

```rust
#[test]
fn read_and_clear_first_header_parity_bit0_compute() {
    let mut tile = Tile::compute(0, 2);
    tile.pkt_handler_status = 0b0001; // First_Header_Parity
    assert_eq!(tile.read_register(0x3FF30), 0b0001);
    // Write-1-to-clear is applied via state::effects in the full path;
    // here assert the read path exposes the bit. The wtc mechanism
    // (effects.rs `&= !(value & 0xF)`) is already covered generically.
}

#[test]
fn read_tlast_error_bit3_memtile() {
    let mut tile = Tile::mem(0, 1);
    tile.pkt_handler_status = 0b1000; // Tlast_Error
    assert_eq!(tile.read_register(0xB0F30), 0b1000);
    assert_eq!(tile.read_register_pure(0xB0F30), 0b1000);
}
```

(Use the same `Tile::compute` / `Tile::mem` constructors the existing
tests in that module use -- match their exact signatures; the existing
`read_returns_pkt_handler_status_memtile` test shows the memtile
constructor and offset.)

- [ ] **Step 2: Run it to verify it fails/passes appropriately**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib pkt_handler_status`
Expected: the two new tests COMPILE and PASS (the read path is already
generic) -- this step confirms the bit-0/bit-3 read-back contract is
locked. If a constructor name differs, fix the test to the real
constructor and re-run.

- [ ] **Step 3: Fix the `tile/mod.rs` misnomer comment**

Replace the `pkt_handler_status` doc block (the lines currently reading
`///   [0] ID_Parity_Error` through `///   [3] Tlast_Error` and the
"We OR a bit in when the reassembler observes" sentence) with:

```rust
    /// Control_Packet_Handler_Status sticky bits (offset 0x3FF30
    /// compute, 0xB0F30 memtile).
    ///
    /// Bit layout per AM025 Tile_Control_Packet_Handler_Status:
    ///   [0] First_Header_Parity_Error  -- detected (stream header)
    ///   [1] Second_Header_Parity_Error -- detected (opcode header)
    ///   [2] SLVERR_On_Access           -- NO detecting path yet
    ///       (successor plan: runtime register-access-error model)
    ///   [3] Tlast_Error                -- detected (write TLAST)
    ///
    /// Sticky bits with write-1-to-clear semantics. The reassembler
    /// returns ReassembleResult::HandlerError for bits 0/1/3; routing.rs
    /// latches them here via PktHandlerError::bit(). Software reads this
    /// register to diagnose, then writes 1 to a bit to clear it.
    /// Compute + memtile only; shim has no packet handler.
```

- [ ] **Step 4: Fix the `effects.rs` misnomer comment**

In `src/device/state/effects.rs` replace the comment line currently
reading
`// Tlast_Error / SLVERR_On_Access / Second_Header_Parity / ID_Parity.`
with:

```rust
        // [3] Tlast_Error / [2] SLVERR_On_Access /
        // [1] Second_Header_Parity_Error / [0] First_Header_Parity_Error
        // (AM025 Tile_Control_Packet_Handler_Status). Write-1-to-clear.
```

(Behavior `tile.pkt_handler_status &= !(value & 0xF)` is unchanged.)

- [ ] **Step 5: Narrow the coverage seed**

In `crates/xdna-archspec/src/coverage/units.rs`, the `control_packets`
seed (the `d("control_packets", ...)` call ~lines 177-180): keep the
verdict `Modeled { completeness: Partial { missing: ... } }` but change
the narrative string and the `missing` string. Replace the existing
narrative sentence
`"... Packet handler status sticky bits + write-1-to-clear; Second_Header_Parity_Error wired, Tlast/SLVERR/ID_Parity not (no detecting path). ..."`
so it reads (keep the surrounding sentences about headers / NPU host
instruction stream intact):

`"... First/Second-header parity and Tlast_Error sticky bits have detecting paths + write-1-to-clear. SLVERR_On_Access deferred: no faithful trigger until the runtime register-access path is repaired (successor plan). Keystone subsystem; was absent from the retired architecture index."`

and change the `missing:` payload from
`"Tlast/SLVERR/ID_Parity packet-handler sticky bits"` to
`"SLVERR_On_Access sticky bit -- needs a register-access-error model (successor plan)"`.

- [ ] **Step 6: Regenerate coverage artifacts**

Run: `cargo run -p xdna-archspec --example gen_coverage_artifacts`
Expected: updates `docs/coverage/aie2/subsystem-index.md` (control_packets
row narrative + `missing` text; verdict stays `Modeled { completeness:
Partial { ... } }`) and `docs/coverage/aie2/implementation-gaps.md`
(the `control_packets: PARTIAL` line text now names SLVERR; queue stays
5 entries). `architecture-index.md` should be unchanged (verdict
unchanged).

- [ ] **Step 7: Run the full suite**

Run: `TMPDIR=/tmp/claude-1000 cargo test --lib`
Expected: PASS for xdna-emu and xdna-archspec (coverage tests green).

- [ ] **Step 8: Commit code + regenerated docs together**

```bash
git add src/device/tile/mod.rs src/device/state/effects.rs \
        src/device/tile/registers.rs \
        crates/xdna-archspec/src/coverage/units.rs \
        docs/coverage/aie2/subsystem-index.md \
        docs/coverage/aie2/implementation-gaps.md \
        docs/coverage/aie2/architecture-index.md
git commit -m "control-packets: fix ID_Parity misnomer + narrow coverage gap

tile/mod.rs + effects.rs comments corrected to the true AM025 4-bit map.
control_packets coverage narrative/missing narrowed to SLVERR-only
(verdict stays Partial); artifacts regenerated in lockstep.

Generated using Claude Code."
```

- [ ] **Step 9: Zero-drift check**

Run: `cargo run -p xdna-archspec --example gen_coverage_artifacts && git diff --stat`
Expected: empty diff (artifacts already committed match a fresh
regeneration), exactly as the Plan-3 staleness gate requires.

---

## Final review (after all tasks)

- [ ] Dispatch a holistic code review over the whole branch delta (spec compliance + cross-task seams: the bit map appears only in `PktHandlerError::bit()`; the `Error` arm sets no bit; SLVERR genuinely absent, not stubbed).
- [ ] Confirm `TMPDIR=/tmp/claude-1000 cargo test --lib` green for xdna-emu and xdna-archspec, zero coverage drift.
- [ ] Use superpowers:finishing-a-development-branch.

## Self-review notes (plan author)

- **Spec coverage:** §0/§1 misnomer + scope -> Task 7. §2 enum -> Task 2. §3.1 First parity -> Task 4. §3.2 Second parity + tighten -> Tasks 3 (tighten) + 5 (detect). §3.3 SLVERR deferred -> not implemented, recorded in Task 2/7 + spec §9. §3.4 Tlast -> Task 6. §4 channel/latch -> Task 3. §5 tests -> per-task Step 1s + Task 1 odd_parity_ok + Task 7 register tests. §6 lockstep -> Task 7 Steps 5-9. §7 files -> Tasks 1-7. No gaps.
- **Placeholder scan:** every code step shows the exact replacement against quoted current code; no TBD/"similar to".
- **Type consistency:** `ReassembleResult::HandlerError(PktHandlerError)`, `PktHandlerError::{FirstHeaderParity,SecondHeaderParity,Tlast}`, `bit()` -> 0x1/0x2/0x8, `odd_parity_ok(u32)->bool`, `latch_pkt_error(&mut Tile, PktHandlerError)` used consistently across Tasks 2-7.
- **Known wrinkle (flagged for implementer/reviewer):** structural `ReassembleResult::Error` is practically unreachable from `feed_word` (2-bit opcode always valid; `beats >= 1`), so the bit-1 tightening in Task 3 is observably inert until Task 5 adds parity detection. This is expected and called out in Task 3's preamble -- not a defect.
