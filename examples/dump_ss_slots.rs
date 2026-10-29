//! Dump stream-switch slave-SLOT config writes from a raw CDO.
//!
//! For the control-packet hang investigation: lists every write that lands in the
//! SS slave-slot region (tile-local offset 0x3F200..0x3F300, the packet-ID
//! matcher) across all tiles, decoding the slot's packet-id/mask/msel/arbiter and
//! Enable bit. This shows which (tile, slot, id) routes the CDO actually
//! programs -- so we can see whether a control-response id (e.g. id=2 on the shim)
//! is present or missing.
//!
//! Usage: cargo run --example dump_ss_slots -- path/to/main_aie_cdo_init.bin

use std::collections::BTreeMap;

use xdna_emu::device::registers::TileAddress;
use xdna_emu::parser::cdo::{find_cdo_offset, Cdo, CdoRaw};

fn main() {
    let mut unknown = 0usize;
    let mut total = 0usize;
    // Flatten every register write across all provided CDO files.
    let mut mem: BTreeMap<u32, u32> = BTreeMap::new();
    for path in std::env::args().skip(1) {
        let data = std::fs::read(&path).expect("read cdo");
        let off = find_cdo_offset(&data).expect("cdo magic");
        let cdo = Cdo::parse(&data[off..]).expect("parse cdo");
        eprintln!("== {path} ==");
        for cmd in cdo.commands() {
            total += 1;
            match cmd {
                CdoRaw::Write { address, value } => {
                    mem.insert(address, value);
                }
                CdoRaw::Write64 { address, value } => {
                    mem.insert(address as u32, value);
                }
                CdoRaw::MaskWrite { address, mask, value } => {
                    let cur = mem.get(&address).copied().unwrap_or(0);
                    mem.insert(address, (cur & !mask) | (value & mask));
                }
                CdoRaw::DmaWrite { address, data } => {
                    for (i, chunk) in data.chunks(4).enumerate() {
                        let mut w = [0u8; 4];
                        w[..chunk.len()].copy_from_slice(chunk);
                        mem.insert(address + (i as u32) * 4, u32::from_le_bytes(w));
                    }
                }
                CdoRaw::Unknown { .. } => unknown += 1,
                _ => {}
            }
        }
    }
    eprintln!("total ops={total} unknown(dropped)={unknown}");

    // Walk the flattened map; report any address whose tile-local offset is in a
    // SS slave-slot region. Slot value: Enable[8], id[28:24], mask[20:16],
    // msel[5:4], arbiter[2:0]. Two distinct slot-config bases:
    //   compute/shim tiles: 0x3F200 (per AM025 core/shim modules)
    //   memtile (row 1):    0xB0200 (per AM025 memory_tile module; ports are
    //                       DMA_0..5, then South_0..5, then North_0..5, stride
    //                       0x10/port with 4 slots/port stride 4)
    println!("Configured SS slots (Enable=1):");
    for (&addr, &v) in mem.iter() {
        let ta = TileAddress::decode(addr);
        let base = if (0x3F200..0x3F600).contains(&ta.offset) {
            0x3F200u32
        } else if (0xB0200..0xB0400).contains(&ta.offset) {
            0xB0200u32
        } else {
            continue;
        };
        if v & 0x100 == 0 {
            continue; // not enabled
        }
        let slot = (ta.offset - base) / 4;
        let id = (v >> 24) & 0x1F;
        let mask = (v >> 16) & 0x1F;
        let msel = (v >> 4) & 0x3;
        let arb = v & 0x7;
        println!(
            "  tile({},{}) off=0x{:05x} slot{slot:2} = 0x{v:08x}  id={id} mask=0x{mask:02x} msel={msel} arb={arb}",
            ta.col, ta.row, ta.offset
        );
    }

    // SS master-port config (0x3F000..0x3F100): master[5:0] = drop-header / which
    // slave feeds it; bit30 packet-enable. Helps map which physical master a slot's
    // msel/arb selects.
    println!("\nConfigured SS masters (enable=1):");
    for (&addr, &v) in mem.iter() {
        let ta = TileAddress::decode(addr);
        if ta.offset < 0x3F000 || ta.offset >= 0x3F100 {
            continue;
        }
        if v & 0x8000_0000 == 0 {
            continue;
        }
        let m = (ta.offset - 0x3F000) / 4;
        println!(
            "  tile({},{}) master{m:2} = 0x{v:08x}{}",
            ta.col,
            ta.row,
            if v & 0x4000_0000 != 0 { " (PKT)" } else { "" }
        );
    }

    // Anything touching the control-packet / controller-id config. The controller
    // id (pkt_id=4) must be programmed into a tile register somewhere; surface any
    // write whose VALUE encodes 4 in a plausible id field, plus known control regs.
    println!("\nAll non-BD/non-SS writes (control-config candidates):");
    for (&addr, &v) in mem.iter() {
        let ta = TileAddress::decode(addr);
        let o = ta.offset;
        // Skip BD regions (0x1D000-0x1D200 shim, 0xA0000-0xA0600 memtile,
        // 0x1D000 compute) and SS region (0x3F000-0x3F900) -- already covered.
        let is_bd = (0x1D000..0x1D300).contains(&o) || (0xA0000..0xA0700).contains(&o);
        let is_ss = (0x3F000..0x3F900).contains(&o);
        if is_bd || is_ss {
            continue;
        }
        println!("  tile({},{}) off=0x{o:05x} = 0x{v:08x}", ta.col, ta.row);
    }
}
