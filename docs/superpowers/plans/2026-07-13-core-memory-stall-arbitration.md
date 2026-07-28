# Core Memory-Bank Arbitration (MEMORY_STALL) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the emulator's inverted core-vs-DMA memory-stall model with faithful per-physical-bank round-robin arbitration, so a lost arbitration stalls the core in the cycle it happens and costs a cycle.

**Architecture:** Correct the physical bank selector; add a pure per-bank round-robin arbiter; reorder the coordinator's per-cycle loop from "commit then observe" to "request -> arbitrate -> commit", withholding the loser (core bundle or DMA channel) for one cycle via the existing stall-retry machinery.

**Tech Stack:** Rust; existing `InterpreterEngine` cycle loop, `CycleAccurateExecutor`, `DmaEngine`, trace event unit.

**Spec:** [`docs/superpowers/specs/2026-07-13-core-memory-stall-arbitration-design.md`](../specs/2026-07-13-core-memory-stall-arbitration-design.md)
**Finding:** [`docs/superpowers/findings/2026-07-13-memory-stall-bank-arbitration.md`](../findings/2026-07-13-memory-stall-bank-arbitration.md)

## Global Constraints

- **Derive, do not calibrate.** No fitted constants. No priority constant installed to hit 220. If faithful round-robin under-produces, report it as a finding.
- **Arbitration is PHYSICAL:** 8 single-port banks, 8 independent round-robin arbiters (AM020 ch.2:164,166; confirmed by HW capture).
- **Bundle granularity:** a conflict on ANY core port stalls the whole datapath (AM020 ch.4:69). Gate the whole bundle; do not split slots.
- **Compute tiles only.** Do NOT change memtile bank geometry (unvalidated).
- `cargo test --lib` must stay green. Baseline on this branch: **3899 passed, 0 failed**.
- Success target is **faithful mechanism + ballpark counts**, not byte-match against the capture.
- No emoji anywhere. Commit messages end with `Generated using Claude Code.`

**Reference HW capture** (`build/experiments/memory-stall-bankcap/events.json`), tiles HW-shifted col0->col1:

| Tile | MEMORY_STALL | CONFLICT bank0 | bank1 | banks2-7 |
|------|---:|---:|---:|---:|
| Producer (1,2) | 1 | 2 | 1 | 0 |
| ConsA (1,3) | 220 | 115 | 109 | 0 |
| ConsB (2,3) | 245 | 128 | 121 | 0 |

---

## File Structure

| File | Responsibility | Action |
|------|----------------|--------|
| `src/device/banking.rs` | Address -> physical bank mapping, per tile-kind layout | Modify (rewrite selector) |
| `src/device/bank_arbiter.rs` | Pure per-bank round-robin arbiter | **Create** |
| `src/interpreter/timing/memory.rs` | `MemoryAccess::bank()` selector | Modify (`:305`) |
| `src/interpreter/execute/cycle_accurate.rs` | Core bank-demand peek (no commit) | Modify (near `:207` `record_memory_access`) |
| `src/device/dma/engine/stepping.rs` | DMA bank-demand peek + withhold transfer | Modify (`:1472,1653,1903,1921,2442`) |
| `src/device/array/dma_ops.rs` | `step_all_dma` gating | Modify (`:216-237`) |
| `src/interpreter/core/interpreter.rs` | `WaitBank` stall status + retry | Modify (`:183,660`) |
| `src/interpreter/engine/coordinator.rs` | Per-cycle phase reorder; event emission | Modify (Phase 2/3/4, `:1418-1552`) |

---

### Task 1: Correct the physical bank mapping

The current selector is a flat 8-way 16-byte interleave across the whole address
space, which scatters one logical bank across all eight physical banks. Replace
with the AM020 layout: four contiguous 16 KB logical banks, each an interleaved
pair of two 8 KB physical banks, alternating every 16 bytes.

**Files:**
- Modify: `src/device/banking.rs`
- Modify: `src/interpreter/timing/memory.rs:300-315`
- Test: inline `#[cfg(test)] mod tests` in `src/device/banking.rs`

**Interfaces:**
- Produces: `BankLayout` enum (`Compute` | `MemTile`), `BankLayout::physical_bank(addr: u32) -> u8`, `banks_for_access(addr: u32, bytes: usize, layout: BankLayout) -> u16`. Tasks 2-6 consume these.

- [ ] **Step 1: Write the failing tests**

Add to `src/device/banking.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    // AM020 ch.2:164 -- 64 KB as eight 8 KB single-port physical banks; every
    // two are interleaved (16-byte granularity) into one 16 KB logical bank.
    // Confirmed by HW: of_q0_rich buffers at 0x400..0x5ff fire CONFLICT_DM_BANK
    // on physical banks 0 and 1 only, near-evenly.
    #[test]
    fn compute_physical_bank_interleaves_pair_every_16_bytes() {
        assert_eq!(BankLayout::Compute.physical_bank(0x0000), 0);
        assert_eq!(BankLayout::Compute.physical_bank(0x0010), 1);
        assert_eq!(BankLayout::Compute.physical_bank(0x0020), 0);
        assert_eq!(BankLayout::Compute.physical_bank(0x0030), 1);
        // within a 16-byte word the bank does not change
        assert_eq!(BankLayout::Compute.physical_bank(0x0004), 0);
        assert_eq!(BankLayout::Compute.physical_bank(0x001C), 1);
    }

    #[test]
    fn compute_logical_banks_are_contiguous_16kb() {
        // logical 0 -> physical {0,1}; logical 1 -> {2,3}; 2 -> {4,5}; 3 -> {6,7}
        assert_eq!(BankLayout::Compute.physical_bank(0x0000), 0);
        assert_eq!(BankLayout::Compute.physical_bank(0x4000), 2);
        assert_eq!(BankLayout::Compute.physical_bank(0x4010), 3);
        assert_eq!(BankLayout::Compute.physical_bank(0x8000), 4);
        assert_eq!(BankLayout::Compute.physical_bank(0xC000), 6);
        assert_eq!(BankLayout::Compute.physical_bank(0xC010), 7);
    }

    #[test]
    fn repro_kernel_buffers_land_in_banks_0_and_1_only() {
        // of_q0_rich consumer buffers: 0x400..0x5ff (ei/eo). HW fired conflicts
        // on banks 0 and 1 ONLY -- banks 2-7 silent.
        let mut seen = 0u16;
        for addr in (0x400u32..0x600).step_by(16) {
            seen |= 1 << BankLayout::Compute.physical_bank(addr);
        }
        assert_eq!(seen, 0b0000_0011, "buffers must occupy exactly banks 0 and 1");
    }

    #[test]
    fn banks_for_access_covers_every_16_byte_word_touched() {
        // a 4-byte scalar access touches one bank
        assert_eq!(banks_for_access(0x400, 4, BankLayout::Compute), 1 << 0);
        // a 32-byte vector access spans two 16-byte words -> two physical banks
        assert_eq!(banks_for_access(0x400, 32, BankLayout::Compute), (1 << 0) | (1 << 1));
        // unaligned access straddling a 16-byte boundary touches both
        assert_eq!(banks_for_access(0x40C, 8, BankLayout::Compute), (1 << 0) | (1 << 1));
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib banking -- --nocapture`
Expected: FAIL — `BankLayout` not defined.

- [ ] **Step 3: Implement the layout**

Rewrite the selector in `src/device/banking.rs`. Keep the existing public
`banks_for_access` name but change its third parameter from `num_banks` to
`BankLayout` (update all call sites in Steps 4-5):

```rust
/// Physical memory-bank layout of a tile's data memory.
///
/// AIE2 compute-tile data memory is 64 KB as eight 8 KB physical banks
/// (512 word x 128-bit, single-port). Every two physical banks are interleaved
/// at 16-byte granularity to form one contiguous 16 KB logical bank, giving the
/// four banks the compiler allocates in (AM020 ch.2:164; AIETargetModel
/// getNumBanks == 4 for compute tiles).
///
/// Arbitration is per PHYSICAL bank -- each has its own round-robin arbiter
/// (AM020 ch.2:166), and the hardware exposes eight CONFLICT_DM_BANK events.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum BankLayout {
    Compute,
    /// MemTile geometry is NOT validated against hardware; preserved as-is.
    MemTile,
    /// Shim tiles have no local data memory banks.
    None,
}

/// Number of physical banks in a compute-tile data memory.
pub const COMPUTE_PHYSICAL_BANKS: u32 = 8;
/// Size of one contiguous logical bank (a pair of interleaved physical banks).
const COMPUTE_LOGICAL_BANK_SHIFT: u32 = 14; // 16 KB
/// Physical banks of a logical pair alternate every 128-bit (16-byte) word.
const COMPUTE_INTERLEAVE_SHIFT: u32 = 4;

impl BankLayout {
    /// Physical bank index for a tile-local byte offset.
    #[inline]
    pub fn physical_bank(&self, addr: u32) -> u8 {
        match self {
            BankLayout::Compute => {
                let logical = (addr >> COMPUTE_LOGICAL_BANK_SHIFT) & 0x3;
                let half = (addr >> COMPUTE_INTERLEAVE_SHIFT) & 0x1;
                (2 * logical + half) as u8
            }
            // Unvalidated: preserve the previous flat interleave for memtiles.
            BankLayout::MemTile => ((addr >> COMPUTE_INTERLEAVE_SHIFT) & 0xF) as u8,
            BankLayout::None => 0,
        }
    }

    /// Number of physical banks this layout arbitrates over.
    #[inline]
    pub fn num_banks(&self) -> u32 {
        match self {
            BankLayout::Compute => COMPUTE_PHYSICAL_BANKS,
            BankLayout::MemTile => 16,
            BankLayout::None => 0,
        }
    }
}

/// Bitmask of every physical bank an access touches.
///
/// An access spans one or more 128-bit words; each word lives in one physical
/// bank, so a wide (vector) or unaligned access can touch several.
#[inline]
pub fn banks_for_access(addr: u32, bytes: usize, layout: BankLayout) -> u16 {
    if bytes == 0 || layout == BankLayout::None {
        return 0;
    }
    let mut mask = 0u16;
    let end = addr.saturating_add(bytes as u32);
    let mut word = addr & !0xF; // 128-bit word containing the first byte
    while word < end {
        mask |= 1 << layout.physical_bank(word);
        word += 16;
    }
    mask
}
```

- [ ] **Step 4: Point `MemoryAccess::bank()` at the same layout**

In `src/interpreter/timing/memory.rs`, replace the hardcoded selector at
`:300-315`:

```rust
    /// Get the physical bank index for this access.
    ///
    /// Physical banks are 8 KB single-port memories; pairs interleave every 16
    /// bytes into the four contiguous 16 KB logical banks the compiler sees
    /// (AM020 ch.2:164). Derived, not hardcoded -- see `device::banking`.
    #[inline]
    pub fn bank(&self) -> u8 {
        crate::device::banking::BankLayout::Compute.physical_bank(self.address)
    }

    /// Get the logical bank index (0-3): the compiler-visible 16 KB bank.
    #[inline]
    pub fn logical_bank(&self) -> u8 {
        self.bank() >> 1
    }
```

- [ ] **Step 5: Update every `banks_for_access` call site to pass a `BankLayout`**

Call sites take `tile.num_banks()` today. Replace with the tile's layout. In
`src/device/tile/mod.rs`, add alongside `num_banks()` (`:793`):

```rust
    /// Physical bank layout of this tile's data memory.
    #[inline]
    pub fn bank_layout(&self) -> crate::device::banking::BankLayout {
        use crate::device::banking::BankLayout;
        match self.kind {
            TileKind::Compute => BankLayout::Compute,
            TileKind::Mem => BankLayout::MemTile,
            _ => BankLayout::None,
        }
    }
```

Then update each caller (compile errors will enumerate them): `tile/mod.rs:804`
(`record_dma_bank_access`), `context.rs:975` (`record_core_bank_access` — pass
the layout instead of `num_banks`), and `stepping.rs:1472,1653,1903,1921,2442`.

- [ ] **Step 6: Run tests**

Run: `cargo test --lib banking -- --nocapture`
Expected: PASS (4 new tests).

Run: `cargo test --lib`
Expected: PASS. **If any pre-existing bank-conflict test now fails, that is
expected** — the old selector was wrong. Inspect each failure: if it asserted
the old `(addr>>4)&7` behavior, update it to the derived layout and note why in
the commit. If it asserts something else, stop and investigate.

- [ ] **Step 7: Commit**

```bash
git add src/device/banking.rs src/device/tile/mod.rs src/interpreter/timing/memory.rs src/interpreter/state/context.rs src/device/dma/engine/stepping.rs
git commit -m "fix(timing): derive physical bank mapping from AM020 layout

The bank selector was a flat (addr>>4)&7 -- an 8-way 16-byte interleave across
the whole address space, which scatters one logical bank across all eight
physical banks. AIE2 compute-tile memory is eight 8 KB single-port physical
banks, paired (interleaved every 16 bytes) into four contiguous 16 KB logical
banks (AM020 ch.2:164; AIETargetModel getNumBanks == 4).

HW-confirmed: of_q0_rich buffers at 0x400-0x5ff fire CONFLICT_DM_BANK on
physical banks 0 and 1 ONLY, near-evenly split -- exactly the derived pairing.

MemTile geometry is unvalidated and left untouched.

Generated using Claude Code."
```

---

### Task 2: The per-bank round-robin arbiter (pure, isolated)

**Files:**
- Create: `src/device/bank_arbiter.rs`
- Modify: `src/device/mod.rs` (add `pub mod bank_arbiter;`)
- Test: inline `#[cfg(test)] mod tests` in `src/device/bank_arbiter.rs`

**Interfaces:**
- Consumes: nothing (pure).
- Produces: `Requester` enum, `BankArbiter::new()`, `BankArbiter::arbitrate(&mut self, demands: &[(Requester, u16)]) -> Arbitration`, `Arbitration { lost: Vec<Requester>, contended_banks: u16 }`. Task 6 consumes these.

- [ ] **Step 1: Write the failing tests**

Create `src/device/bank_arbiter.rs` with tests first:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_contention_everyone_wins() {
        let mut arb = BankArbiter::new();
        let a = arb.arbitrate(&[
            (Requester::Core, 1 << 0),
            (Requester::S2mm(0), 1 << 1),
        ]);
        assert!(a.lost.is_empty());
        assert_eq!(a.contended_banks, 0);
    }

    #[test]
    fn same_bank_collision_grants_exactly_one() {
        let mut arb = BankArbiter::new();
        let a = arb.arbitrate(&[
            (Requester::Core, 1 << 0),
            (Requester::S2mm(0), 1 << 0),
        ]);
        assert_eq!(a.lost.len(), 1, "exactly one requester loses a single bank");
        assert_eq!(a.contended_banks, 1 << 0);
    }

    #[test]
    fn round_robin_alternates_the_winner() {
        // AM020: "round-robin to avoid starving any requester"
        let mut arb = BankArbiter::new();
        let demands = [(Requester::Core, 1 << 0), (Requester::S2mm(0), 1 << 0)];
        let first = arb.arbitrate(&demands).lost;
        let second = arb.arbitrate(&demands).lost;
        assert_ne!(first, second, "the same requester must not lose twice in a row");
    }

    #[test]
    fn a_requester_losing_any_needed_bank_is_reported_lost() {
        let mut arb = BankArbiter::new();
        // Core needs banks 0 and 1; DMA contends only on bank 1.
        let a = arb.arbitrate(&[
            (Requester::Core, (1 << 0) | (1 << 1)),
            (Requester::S2mm(0), 1 << 1),
        ]);
        // Whoever loses bank 1 is reported; contention is only on bank 1.
        assert_eq!(a.contended_banks, 1 << 1);
        assert_eq!(a.lost.len(), 1);
    }

    #[test]
    fn per_bank_arbiters_are_independent() {
        // Each bank has its OWN round-robin pointer (AM020 ch.2:166).
        let mut arb = BankArbiter::new();
        // Contend only bank 0 twice; bank 3's pointer must be untouched.
        let d0 = [(Requester::Core, 1 << 0), (Requester::S2mm(0), 1 << 0)];
        arb.arbitrate(&d0);
        arb.arbitrate(&d0);
        let d3 = [(Requester::Core, 1 << 3), (Requester::S2mm(0), 1 << 3)];
        let a = arb.arbitrate(&d3);
        assert_eq!(a.contended_banks, 1 << 3);
        assert_eq!(a.lost.len(), 1);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --lib bank_arbiter -- --nocapture`
Expected: FAIL — module/type not defined.

- [ ] **Step 3: Implement the arbiter**

Prepend to `src/device/bank_arbiter.rs`:

```rust
//! Per-physical-bank round-robin memory arbiter.
//!
//! AM020 ch.2:166: "Each memory bank has its own arbitrator to arbitrate
//! between all requesters. The memory bank arbitration is round-robin to avoid
//! starving any requester. It handles a new request every clock cycle. When
//! there are multiple requests in the same cycle to the same memory bank, only
//! one request per cycle is allowed to access the memory. The other requesters
//! are stalled for one cycle and the hardware retries the memory request in the
//! next cycle."
//!
//! Arbitration is over PHYSICAL banks (single-port SRAMs), not the four
//! programmer-visible logical banks -- the hardware exposes eight
//! CONFLICT_DM_BANK events, and an HW capture confirmed a single logical bank
//! splitting its conflicts across two independently-counted arbiters.

use super::banking::COMPUTE_PHYSICAL_BANKS;

/// An agent that can request a data-memory bank in a given cycle.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Requester {
    /// The compute core's load/store ports (bundle granularity: a conflict on
    /// any port stalls the whole datapath, AM020 ch.4:69).
    Core,
    /// Stream-to-memory DMA channel.
    S2mm(u8),
    /// Memory-to-stream DMA channel.
    Mm2s(u8),
}

/// Outcome of one cycle of arbitration.
#[derive(Clone, Debug, Default)]
pub struct Arbitration {
    /// Requesters that were denied at least one bank they asked for. They must
    /// hold their request and retry next cycle.
    pub lost: Vec<Requester>,
    /// Bitmask of banks that had more than one requester this cycle. Drives the
    /// CONFLICT_DM_BANK_n trace events.
    pub contended_banks: u16,
}

/// One round-robin arbiter per physical bank.
#[derive(Clone, Debug)]
pub struct BankArbiter {
    /// Per bank: index into the current cycle's demand list that has priority.
    /// Advances past the winner on every grant, so no requester starves.
    rotor: [u8; COMPUTE_PHYSICAL_BANKS as usize],
}

impl Default for BankArbiter {
    fn default() -> Self {
        Self::new()
    }
}

impl BankArbiter {
    pub fn new() -> Self {
        Self { rotor: [0; COMPUTE_PHYSICAL_BANKS as usize] }
    }

    /// Arbitrate one cycle. `demands` is each requester and the bitmask of
    /// physical banks it needs this cycle.
    ///
    /// Grants at most one requester per contended bank, rotating priority. A
    /// requester denied ANY bank it needs is reported in `lost` -- it must stall
    /// and retry the whole request next cycle.
    pub fn arbitrate(&mut self, demands: &[(Requester, u16)]) -> Arbitration {
        let mut out = Arbitration::default();
        let mut denied = [false; 8]; // index into `demands`

        for bank in 0..COMPUTE_PHYSICAL_BANKS as usize {
            let bit = 1u16 << bank;
            // Who wants this bank?
            let wanters: Vec<usize> = demands
                .iter()
                .enumerate()
                .filter(|(_, (_, mask))| mask & bit != 0)
                .map(|(i, _)| i)
                .collect();

            if wanters.len() < 2 {
                continue; // free, or uncontended -- granted, nothing to do
            }
            out.contended_banks |= bit;

            // Round-robin: the first wanter at or after the rotor wins.
            let start = self.rotor[bank] as usize;
            let winner = *wanters
                .iter()
                .find(|&&i| i >= start)
                .unwrap_or(&wanters[0]);

            for &i in &wanters {
                if i != winner && i < denied.len() {
                    denied[i] = true;
                }
            }
            // Advance past the winner so a different requester leads next time.
            self.rotor[bank] = ((winner + 1) % demands.len().max(1)) as u8;
        }

        for (i, (who, _)) in demands.iter().enumerate() {
            if i < denied.len() && denied[i] {
                out.lost.push(*who);
            }
        }
        out
    }
}
```

Add `pub mod bank_arbiter;` to `src/device/mod.rs`.

- [ ] **Step 4: Run tests**

Run: `cargo test --lib bank_arbiter -- --nocapture`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add src/device/bank_arbiter.rs src/device/mod.rs
git commit -m "feat(timing): per-physical-bank round-robin arbiter

Pure arbiter modelling AM020 ch.2:166 -- one round-robin arbitrator per physical
bank, one grant per bank per cycle, losers hold their request and retry next
cycle. A requester denied any bank it needs is reported lost (bundle
granularity: a conflict on any core port stalls the whole datapath, ch.4:69).

Generated using Claude Code."
```

---

### Task 3: Core bank-demand peek (no commit)

The core must be able to say which banks its NEXT bundle needs without executing
it. Address generation is already a pure function of `ctx`
(`MemoryUnit::get_address` / `get_store_address`, `memory/mod.rs:1384,1453`), and
`cycle_accurate.rs:207 record_memory_access` already re-derives addresses without
doing the data movement — this task exposes that as a first-class peek.

**Files:**
- Modify: `src/interpreter/execute/cycle_accurate.rs` (add `peek_bank_demand`)
- Test: inline test in `src/interpreter/execute/cycle_accurate.rs`

**Interfaces:**
- Consumes: `BankLayout`, `banks_for_access` (Task 1).
- Produces: `CycleAccurateExecutor::peek_bank_demand(&self, bundle: &VliwBundle, ctx: &ExecutionContext, layout: BankLayout) -> u16`. Task 6 consumes it.

- [ ] **Step 1: Write the failing test**

```rust
    #[test]
    fn peek_bank_demand_does_not_mutate_context_and_reports_banks() {
        // Build a bundle with a single scalar load from a known local address,
        // mirroring the existing bundle-construction helpers in this module's
        // tests (see the surrounding `mod tests` for the established pattern).
        let (exec, bundle, ctx) = fixture_scalar_load_at(0x0410);
        let before_pc = ctx.pc();
        let before_cycles = ctx.cycles;

        let banks = exec.peek_bank_demand(&bundle, &ctx, BankLayout::Compute);

        assert_eq!(banks, 1 << 1, "0x410 is physical bank 1");
        assert_eq!(ctx.pc(), before_pc, "peek must not advance the PC");
        assert_eq!(ctx.cycles, before_cycles, "peek must not advance the clock");
    }
```

(Use the module's existing test fixture style to build `exec`, `bundle`, `ctx`;
`fixture_scalar_load_at` is a helper you add mirroring the neighbouring tests.)

- [ ] **Step 2: Run to verify it fails**

Run: `cargo test --lib peek_bank_demand -- --nocapture`
Expected: FAIL — method not defined.

- [ ] **Step 3: Implement the peek**

In `src/interpreter/execute/cycle_accurate.rs`, alongside `record_memory_access`:

```rust
    /// Banks this bundle's memory slots will need, WITHOUT executing it.
    ///
    /// Address generation is a pure function of register state, so we can ask
    /// the arbiter for these banks before deciding whether to let the bundle
    /// commit. Returns 0 for a bundle with no local-memory access (it can
    /// always issue).
    pub fn peek_bank_demand(
        &self,
        bundle: &VliwBundle,
        ctx: &ExecutionContext,
        layout: crate::device::banking::BankLayout,
    ) -> u16 {
        use crate::device::banking::banks_for_access;
        use crate::interpreter::execute::memory::{decode_data_address, MemoryQuadrant, MemoryUnit};

        let mut mask = 0u16;
        for op in bundle.memory_ops() {
            let addr = if op.is_store() {
                MemoryUnit::get_store_address(op, ctx)
            } else {
                MemoryUnit::get_address(op, ctx)
            };
            let (quadrant, local_offset) = decode_data_address(addr);
            if quadrant == MemoryQuadrant::Local {
                mask |= banks_for_access(local_offset as u32, op.width_bytes(), layout);
            }
        }
        mask
    }
```

Match `bundle.memory_ops()` / `op.is_store()` / `op.width_bytes()` to the
existing slot-iteration idiom used by `record_memory_access` (`:207-253`); reuse
those exact accessors rather than inventing new ones.

- [ ] **Step 4: Run tests**

Run: `cargo test --lib peek_bank_demand -- --nocapture`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/interpreter/execute/cycle_accurate.rs
git commit -m "feat(timing): non-committing core bank-demand peek

Address generation is already a pure function of register state, so a bundle's
bank demand can be computed before deciding whether to let it commit. This is
the 'request' half of request -> arbitrate -> commit.

Generated using Claude Code."
```

---

### Task 4: DMA bank-demand peek and withheld transfer

The DMA must (a) report the banks it intends to touch this cycle without
transferring, and (b) be able to skip its transfer for a cycle when it loses
arbitration, retrying next cycle unchanged.

**Files:**
- Modify: `src/device/dma/engine/stepping.rs` (`:1472,1653,1903,1921,2442`)
- Modify: `src/device/array/dma_ops.rs` (`step_all_dma`, `:216-237`)
- Test: inline tests in `src/device/dma/engine/stepping.rs`

**Interfaces:**
- Consumes: `BankLayout`, `banks_for_access` (Task 1); `Requester` (Task 2).
- Produces: `DmaEngine::peek_bank_demand(&self, layout) -> Vec<(Requester, u16)>` and `DmaEngine::step_with_denied(&mut self, denied: &[Requester], ...)`. Task 6 consumes them.

- [ ] **Step 1: Write the failing tests**

```rust
    #[test]
    fn dma_peek_does_not_advance_state() {
        let mut eng = fixture_s2mm_engine_mid_transfer();
        let before = eng.clone();
        let demand = eng.peek_bank_demand(BankLayout::Compute);
        assert!(!demand.is_empty(), "an active channel must declare a bank");
        assert_eq!(format!("{:?}", eng), format!("{:?}", before),
                   "peek must not mutate DMA state");
    }

    #[test]
    fn denied_channel_does_not_transfer_and_retries_unchanged() {
        let mut eng = fixture_s2mm_engine_mid_transfer();
        let before_bytes = eng.bytes_transferred();
        eng.step_with_denied(&[Requester::S2mm(0)], /* ..existing step args.. */);
        assert_eq!(eng.bytes_transferred(), before_bytes,
                   "a denied channel must not move data this cycle");

        // Next cycle, ungated, it proceeds normally.
        eng.step_with_denied(&[], /* ..existing step args.. */);
        assert!(eng.bytes_transferred() > before_bytes, "it must retry and progress");
    }
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --lib dma_peek -- --nocapture` and `cargo test --lib denied_channel -- --nocapture`
Expected: FAIL — methods not defined.

- [ ] **Step 3: Implement peek + gate**

In `stepping.rs`, the bank is already computed immediately before each copy
(`:1472,1653,1903,1921,2442`), from descriptor state that does not depend on the
core. Extract that computation into a peek, and gate the copy:

```rust
    /// Banks each active channel intends to touch this cycle, without
    /// transferring. Derived purely from buffer-descriptor state.
    pub fn peek_bank_demand(
        &self,
        layout: crate::device::banking::BankLayout,
    ) -> Vec<(Requester, u16)> {
        let mut out = Vec::new();
        for (ch, s2mm) in self.s2mm_channels() {
            if let Some((offset, bytes)) = s2mm.next_access() {
                out.push((Requester::S2mm(ch), banks_for_access(offset, bytes, layout)));
            }
        }
        for (ch, mm2s) in self.mm2s_channels() {
            if let Some((offset, bytes)) = mm2s.next_access() {
                out.push((Requester::Mm2s(ch), banks_for_access(offset, bytes, layout)));
            }
        }
        out
    }
```

Then in the engine step, take the denied set and skip the transfer for those
channels only — leaving all descriptor state (cursor, byte count, BD pointer)
untouched so the request is retried verbatim next cycle. `next_access()` must be
the same derivation the transfer path uses, so peek and commit cannot disagree.

- [ ] **Step 4: Run tests**

Run: `cargo test --lib dma -- --nocapture`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/device/dma/engine/stepping.rs src/device/array/dma_ops.rs
git commit -m "feat(timing): DMA bank-demand peek and withheld transfer

A DMA channel can now declare the banks it intends to touch this cycle from
descriptor state alone, and can be denied -- skipping the transfer with all
descriptor state untouched, so it retries the identical request next cycle
(AM020 ch.2:166 retry semantics).

Generated using Claude Code."
```

---

### Task 5: `WaitBank` core stall status

A core denied a bank must not execute its bundle this cycle, must cost one
cycle, and must retry the same bundle next cycle — exactly like `WaitLock`.

**Files:**
- Modify: `src/interpreter/core/interpreter.rs` (`:183` `try_resume_stall` entry, `:609-660`)
- Test: inline test in `src/interpreter/core/interpreter.rs`

**Interfaces:**
- Produces: `CoreStatus::WaitBank`, `CoreInterpreter::stall_for_bank(&mut self, ctx: &mut ExecutionContext)`. Task 6 consumes.

- [ ] **Step 1: Write the failing test**

```rust
    #[test]
    fn bank_stall_costs_a_cycle_and_does_not_advance_pc() {
        let (mut core, mut ctx, mut tile) = fixture_core_at_load_bundle();
        let pc_before = ctx.pc();
        let cycles_before = ctx.cycles;

        core.stall_for_bank(&mut ctx);

        assert_eq!(ctx.pc(), pc_before, "a bank-stalled bundle must not retire");
        assert_eq!(ctx.cycles, cycles_before + 1, "the stall must cost one cycle");
        assert_eq!(core.status(), CoreStatus::WaitBank);
    }

    #[test]
    fn bank_stall_clears_and_the_same_bundle_reissues() {
        let (mut core, mut ctx, mut tile) = fixture_core_at_load_bundle();
        let pc = ctx.pc();
        core.stall_for_bank(&mut ctx);
        // Next cycle, ungated: the SAME bundle executes and retires.
        let r = core.step(&mut ctx, &mut tile);
        assert!(matches!(r, StepResult::Continue));
        assert!(ctx.pc() > pc, "the retried bundle must retire");
    }
```

- [ ] **Step 2: Run to verify they fail**

Run: `cargo test --lib bank_stall -- --nocapture`
Expected: FAIL — `WaitBank` / `stall_for_bank` not defined.

- [ ] **Step 3: Implement**

Add `WaitBank` to the `CoreStatus` enum, and mirror the `WaitLock` pattern
(`interpreter.rs:609`):

```rust
    /// Deny this cycle's bundle: the core lost a memory-bank arbitration.
    ///
    /// AM020 ch.2:166 -- a denied requester is "stalled for one cycle and the
    /// hardware retries the memory request in the next cycle". The bundle does
    /// NOT execute, the PC does not advance, and the stall costs a cycle. The
    /// core re-arbitrates from scratch next cycle.
    pub fn stall_for_bank(&mut self, ctx: &mut ExecutionContext) {
        ctx.record_stall(1);
        self.status = CoreStatus::WaitBank;
    }
```

`WaitBank` needs no condition re-check inside `try_resume_stall` (unlike
`WaitLock`): the coordinator re-arbitrates every cycle and simply does not call
`stall_for_bank` when the core wins. So `try_resume_stall` should treat
`WaitBank` as immediately clearable — set `Ready` and return `None` so the
bundle re-issues, and let this cycle's arbitration decide again.

- [ ] **Step 4: Run tests**

Run: `cargo test --lib bank_stall -- --nocapture`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/interpreter/core/interpreter.rs
git commit -m "feat(timing): WaitBank core stall -- a lost arbitration costs a cycle

Mirrors the existing WaitLock/WaitDma per-cycle stall-retry shape: the bundle
does not execute, the PC does not advance, and record_stall(1) charges the cycle
that the bank-conflict model has never charged until now.

Generated using Claude Code."
```

---

### Task 6: Reorder the coordinator loop to request -> arbitrate -> commit

This is the integration task. Replace the retroactive Phase 4 observer with a
real arbitration that runs BEFORE either agent commits.

**Files:**
- Modify: `src/interpreter/engine/coordinator.rs` (Phase 2 `:842`, Phase 3
  `step_data_movement`, Phase 4 `:1418-1552`)

**Interfaces:**
- Consumes: `BankArbiter`, `Requester`, `Arbitration` (Task 2);
  `peek_bank_demand` (Tasks 3, 4); `stall_for_bank` (Task 5).

- [ ] **Step 1: Add a per-tile arbiter to the engine**

In the engine struct, one `BankArbiter` per compute tile (the arbiters are
per-bank per-tile and must persist across cycles — the rotor is state):

```rust
    /// One set of per-physical-bank round-robin arbiters per compute tile.
    bank_arbiters: Vec<BankArbiter>,
```

Initialise one per tile index in the constructor.

- [ ] **Step 2: Restructure `step()`**

Replace the current Phase 2 (cores) / Phase 3 (DMA) / Phase 4 (observe) order
with:

```rust
        // Phase A: DMA declares the banks it intends to touch this cycle.
        //          No transfer yet.
        // Phase B: each ready core declares its next bundle's bank demand.
        //          A core already stalled on a lock/DMA/stream declares nothing.
        // Phase C: per-tile, per-physical-bank round-robin arbitration.
        // Phase D: commit the winners, withhold the losers:
        //            core lost  -> stall_for_bank(ctx)  [no bundle, +1 cycle]
        //            core won   -> step the core normally
        //            DMA chan lost -> withheld this cycle, retries next
        //            DMA chan won  -> transfers
        // Phase E: emit MEMORY_STALL (core loss), CONFLICT_DM_BANK_n (any
        //          contended bank), DMA backpressure/starvation (DMA loss).
```

Per compute tile, build the demand list, arbitrate once, then commit:

```rust
        for tile_idx in compute_tiles {
            let layout = self.device.array.tiles[tile_idx].bank_layout();

            let mut demands: Vec<(Requester, u16)> = Vec::new();
            demands.extend(self.device.array.engines[tile_idx].peek_bank_demand(layout));

            let core_idx = /* core index for this tile */;
            let core_demand = if self.cores[core_idx].is_ready_to_issue() {
                let bundle = self.cores[core_idx].peek_next_bundle();
                self.cores[core_idx]
                    .executor
                    .peek_bank_demand(&bundle, &self.cores[core_idx].context, layout)
            } else {
                0
            };
            if core_demand != 0 {
                demands.push((Requester::Core, core_demand));
            }

            let arb = self.bank_arbiters[tile_idx].arbitrate(&demands);

            let core_lost = arb.lost.contains(&Requester::Core);
            if core_lost {
                self.cores[core_idx]
                    .interpreter
                    .stall_for_bank(&mut self.cores[core_idx].context);
            } else {
                // existing Phase 2 core step
            }

            // existing DMA step, with the denied channels withheld
            let denied: Vec<Requester> =
                arb.lost.iter().copied().filter(|r| *r != Requester::Core).collect();
            // ... step_with_denied(&denied, ...)

            // Phase E emission uses arb.contended_banks and core_lost
        }
```

**Delete** the old retroactive Phase 4 block (`coordinator.rs:1418-1552`) and the
temporary `XDNA_EMU_STALL_DEBUG` census instrumentation added during the
investigation — the arbiter now owns this.

- [ ] **Step 3: Run the full suite**

Run: `cargo test --lib`
Expected: compiles and runs. **Timing-sensitive tests will move** — stalls now
cost cycles. Triage every failure:
- an assertion on a *cycle count* that shifted: expected; update it and record
  the before/after in the commit body.
- an assertion on *correctness* (wrong data, deadlock, hang): **blocker** — stop
  and investigate; do not update the assertion.

- [ ] **Step 4: Commit**

```bash
git add src/interpreter/engine/coordinator.rs
git commit -m "feat(timing): reorder the cycle loop to request -> arbitrate -> commit

The bank-conflict model was retroactive: the core committed its bundle, the DMA
committed its transfer, and only then did a Phase-4 observer notice the
collision -- charging nothing and emitting a trace edge a cycle late. Now each
cycle collects both agents' bank demands BEFORE either commits, runs the
per-physical-bank round-robin arbiter, and withholds the loser for one cycle.

A lost arbitration now stalls the core IN the cycle it happens and COSTS a
cycle, so the MEMORY_STALL edge aligns with the memmod CONFLICT edge exactly as
it does on silicon. Removes the retroactive Phase 4 observer and the temporary
census instrumentation.

Generated using Claude Code."
```

---

### Task 7: Event emission (MEMORY_STALL, CONFLICT_DM_BANK_n, DMA pressure)

**Files:**
- Modify: `src/interpreter/engine/coordinator.rs` (Phase E)

- [ ] **Step 1: Emit from the arbitration result**

- `MEMORY_STALL` (core event 23): on `core_lost`. Keep the existing held-level
  `mem_stall_edge` emission — isolated one-cycle stalls each produce their own
  rising edge, so N discrete stalls decode as N events.
- `CONFLICT_DM_BANK_n` (mem events 77-84): one per bit set in
  `arb.contended_banks`, **regardless of who lost** (hardware raises it on
  contention, not on core-loss).
- `DMA_S2MM_n_MEMORY_BACKPRESSURE` (39/40) on a lost S2MM channel;
  `DMA_MM2S_n_MEMORY_STARVATION` (41/42) on a lost MM2S channel. Resolve the
  hardware IDs through the existing toolchain event mapping (as
  `mem_conflict_dm_bank_hw_id` does) — do not hardcode.

- [ ] **Step 2: Verify the encoding assumption**

Run the census kernel and decode; confirm N one-cycle stalls produce N
MEMORY_STALL events (not one merged span).

Run: `cargo test --lib of_q0_rich_bank_overlap_census -- --ignored --nocapture`

- [ ] **Step 3: Commit**

```bash
git add src/interpreter/engine/coordinator.rs
git commit -m "feat(trace): emit bank-conflict and DMA-pressure events from arbitration

CONFLICT_DM_BANK_n now fires on any contended bank (not only core-loss, matching
hardware), and a denied DMA channel raises S2MM memory-backpressure /
MM2S memory-starvation. Event IDs resolved through the toolchain mapping.

Generated using Claude Code."
```

---

### Task 8: Validate against the hardware capture

**Files:**
- Modify: `src/testing/xclbin_suite.rs` (replace the temporary census test with a
  real assertion test)

**Reference:** `build/experiments/memory-stall-bankcap/events.json`

- [ ] **Step 1: Turn the census probe into an assertion test**

Replace `of_q0_rich_bank_overlap_census` with a test that runs the traced
of_q0_rich in-process and asserts the *mechanism*, per the spec's success
criteria (ballpark, not byte-match):

```rust
    /// The arbitration model must reproduce the HW mechanism: conflicts on the
    /// CONSUMERS (not the producer), confined to physical banks 0/1, with the
    /// consumers' stall counts in the right order of magnitude.
    /// HW reference (build/experiments/memory-stall-bankcap/): producer 1 stall
    /// / 3 conflicts; ConsA 220 stalls / 224 conflicts (banks 0+1); ConsB 245 /
    /// 249. Banks 2-7 silent.
    #[test]
    #[ignore = "requires the traced of_q0_rich build; run explicitly"]
    fn of_q0_rich_reproduces_hw_bank_arbitration() {
        // ... run in-process as the census test did ...
        assert_eq!(conflicts_on_banks(2..=7), 0, "banks 2-7 must be silent");
        assert!(consumer_stalls > 100, "consumers must stall in the right order of magnitude, got {}", consumer_stalls);
        assert!(producer_stalls < 20, "the producer must barely stall, got {}", producer_stalls);
    }
```

- [ ] **Step 2: Run it and record the actual numbers**

Run: `cargo test --lib of_q0_rich_reproduces_hw_bank_arbitration -- --ignored --nocapture`

Record the emulator's producer/consumer stall and per-bank conflict counts
side-by-side with the HW capture.

- [ ] **Step 3: If faithful round-robin under-produces, REPORT — do not fit**

Per the spec: if the consumer count comes out well under ~200 (e.g. ~110), that
is a genuine finding about the arbiter's dynamics (request phasing, requester
count, pointer-advance rule). Write it up in the finding; **do not install a
priority constant to hit 220.**

- [ ] **Step 4: Commit**

```bash
git add src/testing/xclbin_suite.rs docs/superpowers/findings/2026-07-13-memory-stall-bank-arbitration.md
git commit -m "test(timing): assert the bank-arbitration mechanism against the HW capture

Replaces the temporary census probe with a real assertion test: conflicts must
land on the consumers, confined to physical banks 0/1, banks 2-7 silent, with
consumer stalls in the right order of magnitude and the producer barely
stalling. Records the emulator-vs-HW counts in the finding.

Generated using Claude Code."
```

---

### Task 9: Blast-radius assessment and gap registry

**Files:**
- Modify: `docs/fidelity-gaps/core-compute-timing.md`
- Modify: `ROADMAP.md` (if the gap's headline status changes)

- [ ] **Step 1: Full library suite**

Run: `cargo test --lib`
Expected: green. Baseline 3899. Record the new count.

- [ ] **Step 2: Bridge sweep (hardware)**

Only with the NPU idle and no other HW suite running:

Run: `./scripts/emu-bridge-test.sh --chess-only`
Expected: no *correctness* regressions vs the last known-good run. Cycle-count
shifts are expected (that is the point). Record pass/fail/diverge counts against
the previous baseline.

- [ ] **Step 3: Update the gap registry**

Rewrite the `core-compute-timing.md` MEMORY_STALL row: mechanism root-caused
(per-physical-bank round-robin), HW-confirmed, model landed, with the residual
(exact count vs HW, driven by DMA burst timing) recorded as a **separate,
deliberately-unfitted gap** per the "idealised fast memory" design decision.

- [ ] **Step 4: Commit**

```bash
git add docs/fidelity-gaps/core-compute-timing.md ROADMAP.md
git commit -m "docs(gaps): MEMORY_STALL root-caused, modelled, and HW-confirmed

The compute-core MEMORY_STALL gap is closed at the mechanism level: per-physical
-bank round-robin arbitration, stalls charged in the cycle they occur. The
residual count delta (driven by DMA burst timing, deliberately modelled as
idealised fast memory rather than any one silicon's pattern) is recorded as a
separate gap, not fitted away.

Generated using Claude Code."
```

---

## Self-Review

**Spec coverage:** bank-mapping fix (Task 1); arbiter + requesters + physical
granularity (Task 2); core demand peek (Task 3); DMA demand peek + withhold
(Task 4); WaitBank stall + cycle cost (Task 5); per-cycle reorder (Task 6);
event emission incl. encoding verification (Task 7); validation vs capture +
report-don't-fit (Task 8); blast radius + gap registry (Task 9). Out-of-scope
items (memtile, neighbours, cascade, AXI) are explicitly untouched.

**Known integration risk:** Tasks 3, 4 and 6 must match existing accessor names
(`bundle.memory_ops()`, `op.is_store()`, `next_access()`, the core's
ready-to-issue predicate). Reuse the idioms already present at
`cycle_accurate.rs:207-253` and `stepping.rs:1472+` rather than inventing
signatures; adjust the code shown here to the real accessors on contact.
