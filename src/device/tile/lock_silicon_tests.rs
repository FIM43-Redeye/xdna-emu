//! Silicon replay of the lock request probe (Phoenix, 2026-11-08).
//!
//! The probe drove locks from the host over control packets: writes set a
//! lock's Lock_value register, reads returned a register or issued a lock
//! request (a read in the Lock_Request space performs the request and
//! returns its result in bit 0). Three runs covered compute-tile and
//! memtile locks; both tiles answered identically. Each table below is one
//! run on a freshly loaded device, in issue order, with every read's
//! silicon value.
//!
//! What the runs pin down:
//! - Lock_value is a 6-bit unsigned count, 0..=63: a write keeps bits [5:0]
//!   (0xFF reads back 0x3F) and 32..=63 are ordinary positive counts.
//! - Acquire with a negative change c is granted iff value >= -c and then
//!   subtracts -c; a refused acquire changes nothing and sets no flag.
//! - Acquire with a change c >= 0 is granted iff value == c and leaves the
//!   value unchanged.
//! - A release adds its change modulo 64 and is granted even when it wraps;
//!   wrapping above 63 sets the lock's overflow bit, below 0 its underflow
//!   bit.
//! - While either bit is set the lock refuses every acquire and release. A
//!   Lock_value write still lands and does not clear the bit; writing 1 to
//!   the bit clears it and frees the lock.
//!
//! RUN1's labels number locks from 8 (compute) and 40 (memtile); RUN2 and
//! RUN3 number them from 0.

use crate::device::DeviceState;

/// One probe access: `W(row, offset, value)` writes a register, `R(row,
/// offset, silicon)` reads one (or issues a lock request) and checks it.
#[derive(Clone, Copy)]
pub(super) enum ProbeOp {
    W(u8, u32, u32),
    R(u8, u32, u32),
}
use ProbeOp::{R, W};

/// The probe ran in a one-column partition; any application column works.
const COL: u8 = 1;

fn replay(name: &str, ops: &[ProbeOp]) {
    let mut state = DeviceState::new_npu1();
    let mut mismatches = Vec::new();
    for (i, op) in ops.iter().enumerate() {
        match *op {
            // The same entry point a control-packet write takes.
            W(row, offset, value) => state.write_tile_register(COL, row, offset, value),
            R(row, offset, silicon) => {
                let emu = state.array.tile_mut(COL, row).read_register(offset);
                if emu != silicon {
                    mismatches.push(format!(
                        "op {i}: row {row} read 0x{offset:05X}: emu 0x{emu:X}, silicon 0x{silicon:X}"
                    ));
                }
            }
        }
    }
    assert!(mismatches.is_empty(), "{name}: {} mismatches\n{}", mismatches.len(), mismatches.join("\n"));
}

#[test]
fn lock_probe_run1_values_requests_and_wrap() {
    replay("RUN1", RUN1);
}

#[test]
fn lock_probe_run2_acquire_modes_and_bounds() {
    replay("RUN2", RUN2);
}

#[test]
fn lock_probe_run3_flags_freeze_the_lock() {
    replay("RUN3", RUN3);
}

pub(super) const RUN1: &[ProbeOp] = &[
    W(2, 0x1F080, 0x28),
    R(2, 0x1F080, 0x28), // L0 init 40 -> value
    R(2, 0x423FC, 0x1),  // L0 acq chg-1 (on 40) result
    R(2, 0x1F080, 0x27), // L0 value after
    R(2, 0x42364, 0x1),  // L0 acq chg-39 (on 39) result
    R(2, 0x1F080, 0x0),  // L0 value after
    W(2, 0x1F090, 0x3F),
    R(2, 0x1F090, 0x3F),  // L1 init 63 -> value
    R(2, 0x42404, 0x1),   // L1 rel +1 (on 63) result
    R(2, 0x1F090, 0x0),   // L1 value after
    R(2, 0x1F120, 0x200), // overflow reg after L1
    W(2, 0x1F0A0, 0x3C),
    R(2, 0x42828, 0x1),   // L2 rel +10 (on 60) result
    R(2, 0x1F0A0, 0x6),   // L2 value after
    R(2, 0x1F120, 0x600), // overflow reg after L2
    W(2, 0x1F0B0, 0x3),
    R(2, 0x42DEC, 0x1),   // L3 rel -5 (on 3) result
    R(2, 0x1F0B0, 0x3E),  // L3 value after
    R(2, 0x1F128, 0x800), // underflow reg after L3
    W(2, 0x1F0C0, 0x3),
    R(2, 0x433EC, 0x0),   // L4 acq chg-5 (on 3) result
    R(2, 0x1F0C0, 0x3),   // L4 value after
    R(2, 0x1F128, 0x800), // underflow reg after L4
    W(2, 0x1F0D0, 0xFF),
    R(2, 0x1F0D0, 0x3F), // L5 write 0xFF -> value
    W(2, 0x1F0E0, 0x20),
    R(2, 0x43BFC, 0x1),  // L6 acq chg-1 (on 32) result
    R(2, 0x1F0E0, 0x1F), // L6 value after
    W(2, 0x1F0F0, 0x0),
    R(2, 0x43DFC, 0x1),    // L7 rel -1 (on 0) result
    R(2, 0x1F0F0, 0x3F),   // L7 value after
    R(2, 0x1F128, 0x8800), // underflow reg after L7
    W(2, 0x1F090, 0xA),
    R(2, 0x42608, 0x0), // L1 acq chg+2 (on 10) result
    R(2, 0x1F090, 0xA), // L1 value after
    W(2, 0x1F090, 0x2),
    R(2, 0x42608, 0x0), // L1 acq chg+2 (on 2) result
    R(2, 0x1F090, 0x2), // L1 value after
    W(2, 0x1F0A0, 0x5),
    R(2, 0x42A00, 0x0), // L2 acq chg0 (on 5) result
    R(2, 0x1F0A0, 0x5), // L2 value after
    W(2, 0x1F0A0, 0x0),
    R(2, 0x42A00, 0x0), // L2 acq chg0 (on 0) result
    R(2, 0x1F0A0, 0x0), // L2 value after
    W(2, 0x1F0B0, 0x3F),
    R(2, 0x42F60, 0x0),  // L3 acq chg-40 (on 63) result
    R(2, 0x1F0B0, 0x3F), // L3 value after
    W(2, 0x1F0C0, 0x0),
    R(2, 0x430A0, 0x1),    // L4 rel +40 (on 0) result
    R(2, 0x1F0C0, 0x28),   // L4 value after
    R(2, 0x43078, 0x1),    // L4 rel +30 (on 40) result
    R(2, 0x1F0C0, 0x6),    // L4 value after
    R(2, 0x1F120, 0x1600), // overflow reg after L4
    W(1, 0xC0280, 0x28),
    R(1, 0xC0280, 0x28), // L0 init 40 -> value
    R(1, 0xDA3FC, 0x1),  // L0 acq chg-1 (on 40) result
    R(1, 0xC0280, 0x27), // L0 value after
    R(1, 0xDA364, 0x1),  // L0 acq chg-39 (on 39) result
    R(1, 0xC0280, 0x0),  // L0 value after
    W(1, 0xC0290, 0x3F),
    R(1, 0xC0290, 0x3F),  // L1 init 63 -> value
    R(1, 0xDA404, 0x1),   // L1 rel +1 (on 63) result
    R(1, 0xC0290, 0x0),   // L1 value after
    R(1, 0xC0424, 0x200), // overflow reg after L1
    W(1, 0xC02A0, 0x3C),
    R(1, 0xDA828, 0x1),   // L2 rel +10 (on 60) result
    R(1, 0xC02A0, 0x6),   // L2 value after
    R(1, 0xC0424, 0x600), // overflow reg after L2
    W(1, 0xC02B0, 0x3),
    R(1, 0xDADEC, 0x1),   // L3 rel -5 (on 3) result
    R(1, 0xC02B0, 0x3E),  // L3 value after
    R(1, 0xC042C, 0x800), // underflow reg after L3
    W(1, 0xC02C0, 0x3),
    R(1, 0xDB3EC, 0x0),   // L4 acq chg-5 (on 3) result
    R(1, 0xC02C0, 0x3),   // L4 value after
    R(1, 0xC042C, 0x800), // underflow reg after L4
    W(1, 0xC02D0, 0xFF),
    R(1, 0xC02D0, 0x3F), // L5 write 0xFF -> value
    W(1, 0xC02E0, 0x20),
    R(1, 0xDBBFC, 0x1),  // L6 acq chg-1 (on 32) result
    R(1, 0xC02E0, 0x1F), // L6 value after
    W(1, 0xC02F0, 0x0),
    R(1, 0xDBDFC, 0x1),    // L7 rel -1 (on 0) result
    R(1, 0xC02F0, 0x3F),   // L7 value after
    R(1, 0xC042C, 0x8800), // underflow reg after L7
    W(1, 0xC0290, 0xA),
    R(1, 0xDA608, 0x0), // L1 acq chg+2 (on 10) result
    R(1, 0xC0290, 0xA), // L1 value after
    W(1, 0xC0290, 0x2),
    R(1, 0xDA608, 0x0), // L1 acq chg+2 (on 2) result
    R(1, 0xC0290, 0x2), // L1 value after
    W(1, 0xC02A0, 0x5),
    R(1, 0xDAA00, 0x0), // L2 acq chg0 (on 5) result
    R(1, 0xC02A0, 0x5), // L2 value after
    W(1, 0xC02A0, 0x0),
    R(1, 0xDAA00, 0x0), // L2 acq chg0 (on 0) result
    R(1, 0xC02A0, 0x0), // L2 value after
    W(1, 0xC02B0, 0x3F),
    R(1, 0xDAF60, 0x0),  // L3 acq chg-40 (on 63) result
    R(1, 0xC02B0, 0x3F), // L3 value after
    W(1, 0xC02C0, 0x0),
    R(1, 0xDB0A0, 0x1),    // L4 rel +40 (on 0) result
    R(1, 0xC02C0, 0x28),   // L4 value after
    R(1, 0xDB078, 0x1),    // L4 rel +30 (on 40) result
    R(1, 0xC02C0, 0x6),    // L4 value after
    R(1, 0xC0424, 0x1600), // overflow reg after L4
];

pub(super) const RUN2: &[ProbeOp] = &[
    W(2, 0x1F120, 0xFFFFFFFF),
    W(2, 0x1F128, 0xFFFFFFFF),
    R(2, 0x1F120, 0x0), // flag reg after clear
    R(2, 0x1F128, 0x0), // flag reg after clear
    W(2, 0x1F000, 0x2),
    R(2, 0x40208, 0x1), // L0=2 acq chg+2 result
    R(2, 0x1F000, 0x2), // L0 value after
    W(2, 0x1F010, 0xA),
    R(2, 0x40608, 0x0), // L1=10 acq chg+2 result
    R(2, 0x1F010, 0xA), // L1 value after
    W(2, 0x1F020, 0x0),
    R(2, 0x40A00, 0x1), // L2=0 acq chg+0 result
    R(2, 0x1F020, 0x0), // L2 value after
    W(2, 0x1F030, 0x5),
    R(2, 0x40E00, 0x0), // L3=5 acq chg+0 result
    R(2, 0x1F030, 0x5), // L3 value after
    W(2, 0x1F040, 0x1),
    R(2, 0x41204, 0x1), // L4=1 acq chg+1 result
    R(2, 0x1F040, 0x1), // L4 value after
    W(2, 0x1F050, 0x3F),
    R(2, 0x41760, 0x1),  // L5=63 acq chg-40 result
    R(2, 0x1F050, 0x17), // L5 value after
    W(2, 0x1F060, 0x3F),
    R(2, 0x41B80, 0x1),  // L6=63 acq chg-32 result
    R(2, 0x1F060, 0x1F), // L6 value after
    W(2, 0x1F070, 0x3F),
    R(2, 0x41F84, 0x1),  // L7=63 acq chg-31 result
    R(2, 0x1F070, 0x20), // L7 value after
    W(2, 0x1F080, 0x3F),
    R(2, 0x4237C, 0x1),  // L8=63 acq chg-33 result
    R(2, 0x1F080, 0x1E), // L8 value after
    W(2, 0x1F090, 0x28),
    R(2, 0x42760, 0x1), // L9=40 acq chg-40 result
    R(2, 0x1F090, 0x0), // L9 value after
    W(2, 0x1F0A0, 0x20),
    R(2, 0x42B80, 0x1), // L10=32 acq chg-32 result
    R(2, 0x1F0A0, 0x0), // L10 value after
    W(2, 0x1F0B0, 0x3F),
    R(2, 0x42F04, 0x1), // L11=63 acq chg-63 result
    R(2, 0x1F0B0, 0x0), // L11 value after
    W(2, 0x1F0C0, 0x3F),
    R(2, 0x43300, 0x0),  // L12=63 acq chg-64 result
    R(2, 0x1F0C0, 0x3F), // L12 value after
    W(2, 0x1F0D0, 0x21),
    R(2, 0x4377C, 0x1), // L13=33 acq chg-33 result
    R(2, 0x1F0D0, 0x0), // L13 value after
    R(2, 0x1F120, 0x0), // flag reg after acquire tests
    R(2, 0x1F128, 0x0), // flag reg after acquire tests
    W(2, 0x1F0E0, 0x3),
    R(2, 0x439EC, 0x1),  // L14=3 rel -5 result
    R(2, 0x1F0E0, 0x3E), // L14 value after
    R(2, 0x43BFC, 0x0),  // L14 (underflow flag set) acq -1 result
    R(2, 0x1F0E0, 0x3E), // L14 value after
    W(2, 0x1F0F0, 0x3F),
    R(2, 0x43C04, 0x1),    // L15=63 rel +1 result
    R(2, 0x1F0F0, 0x0),    // L15 value after
    R(2, 0x43C14, 0x0),    // L15 (overflow flag set) rel +5 result
    R(2, 0x1F0F0, 0x0),    // L15 value after
    R(2, 0x43FFC, 0x0),    // L15 (overflow flag set) acq -1 result
    R(2, 0x1F0F0, 0x0),    // L15 value after
    R(2, 0x1F120, 0x8000), // flag reg after flag tests
    R(2, 0x1F128, 0x4000), // flag reg after flag tests
    W(1, 0xC0420, 0xFFFFFFFF),
    W(1, 0xC0424, 0xFFFFFFFF),
    W(1, 0xC0428, 0xFFFFFFFF),
    W(1, 0xC042C, 0xFFFFFFFF),
    R(1, 0xC0420, 0x0), // flag reg after clear
    R(1, 0xC0424, 0x0), // flag reg after clear
    R(1, 0xC0428, 0x0), // flag reg after clear
    R(1, 0xC042C, 0x0), // flag reg after clear
    W(1, 0xC0000, 0x2),
    R(1, 0xD0208, 0x1), // L0=2 acq chg+2 result
    R(1, 0xC0000, 0x2), // L0 value after
    W(1, 0xC0010, 0xA),
    R(1, 0xD0608, 0x0), // L1=10 acq chg+2 result
    R(1, 0xC0010, 0xA), // L1 value after
    W(1, 0xC0020, 0x0),
    R(1, 0xD0A00, 0x1), // L2=0 acq chg+0 result
    R(1, 0xC0020, 0x0), // L2 value after
    W(1, 0xC0030, 0x5),
    R(1, 0xD0E00, 0x0), // L3=5 acq chg+0 result
    R(1, 0xC0030, 0x5), // L3 value after
    W(1, 0xC0040, 0x1),
    R(1, 0xD1204, 0x1), // L4=1 acq chg+1 result
    R(1, 0xC0040, 0x1), // L4 value after
    W(1, 0xC0050, 0x3F),
    R(1, 0xD1760, 0x1),  // L5=63 acq chg-40 result
    R(1, 0xC0050, 0x17), // L5 value after
    W(1, 0xC0060, 0x3F),
    R(1, 0xD1B80, 0x1),  // L6=63 acq chg-32 result
    R(1, 0xC0060, 0x1F), // L6 value after
    W(1, 0xC0070, 0x3F),
    R(1, 0xD1F84, 0x1),  // L7=63 acq chg-31 result
    R(1, 0xC0070, 0x20), // L7 value after
    W(1, 0xC0080, 0x3F),
    R(1, 0xD237C, 0x1),  // L8=63 acq chg-33 result
    R(1, 0xC0080, 0x1E), // L8 value after
    W(1, 0xC0090, 0x28),
    R(1, 0xD2760, 0x1), // L9=40 acq chg-40 result
    R(1, 0xC0090, 0x0), // L9 value after
    W(1, 0xC00A0, 0x20),
    R(1, 0xD2B80, 0x1), // L10=32 acq chg-32 result
    R(1, 0xC00A0, 0x0), // L10 value after
    W(1, 0xC00B0, 0x3F),
    R(1, 0xD2F04, 0x1), // L11=63 acq chg-63 result
    R(1, 0xC00B0, 0x0), // L11 value after
    W(1, 0xC00C0, 0x3F),
    R(1, 0xD3300, 0x0),  // L12=63 acq chg-64 result
    R(1, 0xC00C0, 0x3F), // L12 value after
    W(1, 0xC00D0, 0x21),
    R(1, 0xD377C, 0x1), // L13=33 acq chg-33 result
    R(1, 0xC00D0, 0x0), // L13 value after
    R(1, 0xC0420, 0x0), // flag reg after acquire tests
    R(1, 0xC0424, 0x0), // flag reg after acquire tests
    R(1, 0xC0428, 0x0), // flag reg after acquire tests
    R(1, 0xC042C, 0x0), // flag reg after acquire tests
    W(1, 0xC00E0, 0x3),
    R(1, 0xD39EC, 0x1),  // L14=3 rel -5 result
    R(1, 0xC00E0, 0x3E), // L14 value after
    R(1, 0xD3BFC, 0x0),  // L14 (underflow flag set) acq -1 result
    R(1, 0xC00E0, 0x3E), // L14 value after
    W(1, 0xC00F0, 0x3F),
    R(1, 0xD3C04, 0x1),    // L15=63 rel +1 result
    R(1, 0xC00F0, 0x0),    // L15 value after
    R(1, 0xD3C14, 0x0),    // L15 (overflow flag set) rel +5 result
    R(1, 0xC00F0, 0x0),    // L15 value after
    R(1, 0xD3FFC, 0x0),    // L15 (overflow flag set) acq -1 result
    R(1, 0xC00F0, 0x0),    // L15 value after
    R(1, 0xC0420, 0x8000), // flag reg after flag tests
    R(1, 0xC0424, 0x0),    // flag reg after flag tests
    R(1, 0xC0428, 0x4000), // flag reg after flag tests
    R(1, 0xC042C, 0x0),    // flag reg after flag tests
];

pub(super) const RUN3: &[ProbeOp] = &[
    W(2, 0x1F120, 0xFFFFFFFF),
    W(2, 0x1F128, 0xFFFFFFFF),
    W(2, 0x1F000, 0x3),
    R(2, 0x401EC, 0x1),  // L0 =3 rel -5 result
    R(2, 0x1F000, 0x3E), // L0 value after
    W(2, 0x1F128, 0x1),
    R(2, 0x1F128, 0x0),  // unf after clearing bit 0
    R(2, 0x403FC, 0x1),  // L0 (cleared) acq -1 result
    R(2, 0x1F000, 0x3D), // L0 value after
    W(2, 0x1F010, 0x3F),
    R(2, 0x40404, 0x1), // L1 =63 rel +1 result
    R(2, 0x1F010, 0x0), // L1 value after
    W(2, 0x1F120, 0x2),
    R(2, 0x1F120, 0x0), // ovf after clearing bit 1
    R(2, 0x40414, 0x1), // L1 (cleared) rel +5 result
    R(2, 0x1F010, 0x5), // L1 value after
    W(2, 0x1F020, 0x3F),
    R(2, 0x40804, 0x1), // L2 =63 rel +1 result
    R(2, 0x1F020, 0x0), // L2 value after
    W(2, 0x1F128, 0x4),
    R(2, 0x1F120, 0x4), // ovf after clearing unf bit 2
    R(2, 0x40804, 0x0), // L2 (ovf still set) rel +1 result
    R(2, 0x1F020, 0x0), // L2 value after
    W(2, 0x1F030, 0x3),
    R(2, 0x40DEC, 0x1),  // L3 =3 rel -5 result
    R(2, 0x1F030, 0x3E), // L3 value after
    R(2, 0x40C04, 0x0),  // L3 (unf set) rel +1 result
    R(2, 0x1F030, 0x3E), // L3 value after
    W(2, 0x1F040, 0x3F),
    R(2, 0x41004, 0x1), // L4 =63 rel +1 result
    R(2, 0x1F040, 0x0), // L4 value after
    R(2, 0x41200, 0x0), // L4 (ovf set) acq +0 result
    R(2, 0x1F040, 0x0), // L4 value after
    W(2, 0x1F050, 0x5),
    R(2, 0x417F4, 0x1),  // L5 =5 acq -3 result
    R(2, 0x1F050, 0x2),  // L5 value after
    R(2, 0x417F4, 0x0),  // L5 (=2) acq -3 result
    R(2, 0x1F050, 0x2),  // L5 value after
    R(2, 0x1F120, 0x14), // ovf final
    R(2, 0x1F128, 0x8),  // unf final
    W(2, 0x1F060, 0x3F),
    R(2, 0x41804, 0x1), // L6 =63 rel +1 result
    R(2, 0x1F060, 0x0), // L6 value after
    W(2, 0x1F060, 0x7),
    R(2, 0x1F120, 0x54), // ovf after value write
    R(2, 0x41BFC, 0x0),  // L6 (=7, ovf set) acq -1 result
    R(2, 0x1F060, 0x7),  // L6 value after
    W(1, 0xC0420, 0xFFFFFFFF),
    W(1, 0xC0424, 0xFFFFFFFF),
    W(1, 0xC0428, 0xFFFFFFFF),
    W(1, 0xC042C, 0xFFFFFFFF),
    W(1, 0xC0000, 0x3),
    R(1, 0xD01EC, 0x1),  // L0 =3 rel -5 result
    R(1, 0xC0000, 0x3E), // L0 value after
    W(1, 0xC0428, 0x1),
    R(1, 0xC0428, 0x0),  // unf after clearing bit 0
    R(1, 0xD03FC, 0x1),  // L0 (cleared) acq -1 result
    R(1, 0xC0000, 0x3D), // L0 value after
    W(1, 0xC0010, 0x3F),
    R(1, 0xD0404, 0x1), // L1 =63 rel +1 result
    R(1, 0xC0010, 0x0), // L1 value after
    W(1, 0xC0420, 0x2),
    R(1, 0xC0420, 0x0), // ovf after clearing bit 1
    R(1, 0xD0414, 0x1), // L1 (cleared) rel +5 result
    R(1, 0xC0010, 0x5), // L1 value after
    W(1, 0xC0020, 0x3F),
    R(1, 0xD0804, 0x1), // L2 =63 rel +1 result
    R(1, 0xC0020, 0x0), // L2 value after
    W(1, 0xC0428, 0x4),
    R(1, 0xC0420, 0x4), // ovf after clearing unf bit 2
    R(1, 0xD0804, 0x0), // L2 (ovf still set) rel +1 result
    R(1, 0xC0020, 0x0), // L2 value after
    W(1, 0xC0030, 0x3),
    R(1, 0xD0DEC, 0x1),  // L3 =3 rel -5 result
    R(1, 0xC0030, 0x3E), // L3 value after
    R(1, 0xD0C04, 0x0),  // L3 (unf set) rel +1 result
    R(1, 0xC0030, 0x3E), // L3 value after
    W(1, 0xC0040, 0x3F),
    R(1, 0xD1004, 0x1), // L4 =63 rel +1 result
    R(1, 0xC0040, 0x0), // L4 value after
    R(1, 0xD1200, 0x0), // L4 (ovf set) acq +0 result
    R(1, 0xC0040, 0x0), // L4 value after
    W(1, 0xC0050, 0x5),
    R(1, 0xD17F4, 0x1),  // L5 =5 acq -3 result
    R(1, 0xC0050, 0x2),  // L5 value after
    R(1, 0xD17F4, 0x0),  // L5 (=2) acq -3 result
    R(1, 0xC0050, 0x2),  // L5 value after
    R(1, 0xC0420, 0x14), // ovf final
    R(1, 0xC0428, 0x8),  // unf final
    W(1, 0xC0060, 0x3F),
    R(1, 0xD1804, 0x1), // L6 =63 rel +1 result
    R(1, 0xC0060, 0x0), // L6 value after
    W(1, 0xC0060, 0x7),
    R(1, 0xC0420, 0x54), // ovf after value write
    R(1, 0xD1BFC, 0x0),  // L6 (=7, ovf set) acq -1 result
    R(1, 0xC0060, 0x7),  // L6 value after
];
