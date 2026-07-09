//! Verified external-stimulus model for the firmware boot path.
//!
//! **Collapse-to-bit3 (2026-07-08).** The ONLY external stimulus the boot path is
//! verified (against executed code) to wait on is the per-column readiness bit.
//! `FUN_00008c68` reads the status byte at `[0xf9e0 + col*0x60]` and gates on
//! bit3 (`L8ui a9,[a8]; Bbci a9,3,<skip>` at `0x8c88`/`0x8c8b`): bit3 CLEAR skips
//! the column, bit3 SET services it. On real silicon an external power-up agent
//! sets that bit when the column comes up. This models exactly that, and nothing
//! else -- zero-latency column power-up.
//!
//! The prior model (a "column-power command descriptor" at `0xfae0` with
//! `{valid,colmask,target}` fields, an i2x mailbox completion, and a
//! `[target+0x30]` done-flag writeback) was a MISREAD: `FUN_00008620` is a
//! data-TLB/cache remap routine, `0xc530` is a generic critical-section IPC
//! message-post primitive (28 callers), and `0x9040` is a garbage address from a
//! windowed call that skipped a mask-init -- not a task pointer. See the
//! collapse-to-bit3 audit in
//! `docs/superpowers/findings/2026-07-08-boot-wake-unreached-breach.md`. Only the
//! bit3 poll survived verification; everything else is deleted.

use super::mmio::Bus;

/// Per-column status byte base. Verified: `FUN_00008c68` reads `[0xf9e0+col*0x60]`
/// as a byte (`L8ui` at `0x8c88`).
const COL_STATUS_BASE: u32 = 0xf9e0;
/// Stride between adjacent per-column status bytes (`Addi a8,a8,96` at `0x8cae`).
const COL_STATUS_STRIDE: u32 = 0x60;
/// The readiness bit the firmware gates on (`Bbci a9,3` at `0x8c8b`).
const COL_READY_BIT: u8 = 0x08;
/// Columns the firmware polls. Verified: a natural boot reads exactly three status
/// bytes -- `0xf9e0`, `0xfa40`, `0xfaa0` -- from the poll PC `0x8c88`; satisfying
/// them reveals no fourth. (The deleted descriptor's `colmask=0xf` claimed four.)
const NUM_COLS: u32 = 3;

/// The one verified boot agent: assert the per-column readiness bit (bit3) for the
/// columns the firmware polls, modelling a zero-latency external column power-up.
/// Level-based (re-asserted every tick) because the firmware clears the bit after
/// servicing a column (`And a9,a9,a6; S8i` at `0x8ca8`) and real power stays up.
#[derive(Default)]
pub struct HostMailbox {
    enabled: bool,
    /// Diagnostic: number of times the agent set a bit that had been clear (rising
    /// edges), and the last column status address it set.
    ready_edges: u32,
    last_col: Option<u32>,
}

impl HostMailbox {
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable the model for the boot-to-idle path. A no-op until enabled, so it
    /// does not perturb firmware tests that step the CPU for other reasons.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// One step: assert bit3 on each polled column's status byte. No-op when
    /// disabled.
    pub fn tick(&mut self, bus: &mut Bus) {
        if !self.enabled {
            return;
        }
        for col in 0..NUM_COLS {
            let sb = COL_STATUS_BASE + col * COL_STATUS_STRIDE;
            let cur = bus.data_load8(sb);
            if cur & COL_READY_BIT == 0 {
                bus.data_store8(sb, u32::from(cur | COL_READY_BIT));
                self.ready_edges += 1;
                self.last_col = Some(sb);
            }
        }
    }

    /// Diagnostic: `(readiness-bit rising edges asserted, last column addr set)`.
    /// Kept in the shape the boot probes expect; the second element is the column
    /// status address rather than a former `(colmask,target)` pair.
    pub fn column_stats(&self) -> (u32, Option<u32>) {
        (self.ready_edges, self.last_col)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_tick_is_a_noop() {
        let mut bus = Bus::new(vec![]);
        let mut hm = HostMailbox::new(); // not enabled
        hm.tick(&mut bus);
        for col in 0..NUM_COLS {
            let sb = COL_STATUS_BASE + col * COL_STATUS_STRIDE;
            assert_eq!(bus.data_load8(sb), 0, "col {col} untouched while disabled");
        }
        assert_eq!(hm.column_stats(), (0, None));
    }

    #[test]
    fn sets_bit3_on_each_polled_column() {
        let mut bus = Bus::new(vec![]);
        let mut hm = HostMailbox::new();
        hm.enable();
        hm.tick(&mut bus);
        for col in 0..NUM_COLS {
            let sb = COL_STATUS_BASE + col * COL_STATUS_STRIDE;
            assert_eq!(bus.data_load8(sb) & COL_READY_BIT, COL_READY_BIT, "col {col} bit3 set");
        }
        // Exactly the three polled columns, no fourth.
        let fourth = COL_STATUS_BASE + NUM_COLS * COL_STATUS_STRIDE;
        assert_eq!(bus.data_load8(fourth), 0, "no fourth column touched");
        assert_eq!(hm.column_stats().0, NUM_COLS, "one rising edge per column");
    }

    #[test]
    fn preserves_other_bits_in_the_status_byte() {
        let mut bus = Bus::new(vec![]);
        bus.data_store8(COL_STATUS_BASE, 0x05); // bits 0 and 2 already set
        let mut hm = HostMailbox::new();
        hm.enable();
        hm.tick(&mut bus);
        assert_eq!(bus.data_load8(COL_STATUS_BASE), 0x0d, "bit3 added, bits 0/2 kept");
    }

    #[test]
    fn level_reassert_after_firmware_clears_the_bit() {
        let mut bus = Bus::new(vec![]);
        let mut hm = HostMailbox::new();
        hm.enable();
        hm.tick(&mut bus);
        // Firmware services the column and clears bit3 (as at 0x8ca8/0x8cab).
        bus.data_store8(COL_STATUS_BASE, 0);
        hm.tick(&mut bus);
        assert_eq!(bus.data_load8(COL_STATUS_BASE) & COL_READY_BIT, COL_READY_BIT, "bit3 re-asserted");
    }

    #[test]
    fn idempotent_when_bit_already_set() {
        let mut bus = Bus::new(vec![]);
        let mut hm = HostMailbox::new();
        hm.enable();
        hm.tick(&mut bus);
        let edges_after_first = hm.column_stats().0;
        hm.tick(&mut bus); // bits already set -> no new rising edges
        assert_eq!(hm.column_stats().0, edges_after_first, "no double-count when already set");
    }
}
