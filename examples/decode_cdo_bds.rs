//! Standalone CDO BD decoder.
//!
//! Expands a raw CDO (`main_aie_cdo_init.bin`) into a flat address->value
//! register map, then decodes the DMA BD registers for every tile, printing
//! buffer length and the dim0/dim1/dim2/iteration stepsize+wrap fields plus
//! zero-padding fields. This reads the binary directly (no BD-struct
//! interpretation) so it is an independent check of what the hardware is
//! actually told to do.
//!
//! Usage: cargo run --example decode_cdo_bds -- path/to/main_aie_cdo_init.bin

use std::collections::BTreeMap;

use xdna_emu::parser::cdo::{find_cdo_offset, Cdo, CdoRaw};
use xdna_emu::device::registers::TileAddress;

fn field(v: u32, msb: u32, lsb: u32) -> u32 {
    let width = msb - lsb + 1;
    let mask = if width == 32 {
        u32::MAX
    } else {
        (1u32 << width) - 1
    };
    (v >> lsb) & mask
}

fn main() {
    let path = std::env::args().nth(1).expect("usage: decode_cdo_bds <cdo.bin>");
    let data = std::fs::read(&path).expect("read cdo");
    let off = find_cdo_offset(&data).expect("cdo magic");
    let cdo = Cdo::parse(&data[off..]).expect("parse cdo");

    // Flatten every register write into a sparse address->value map.
    let mut mem: BTreeMap<u32, u32> = BTreeMap::new();
    for cmd in cdo.commands() {
        match cmd {
            CdoRaw::Write { address, value } => {
                mem.insert(address, value);
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
            _ => {}
        }
    }

    // BD register bases: memtile (row 1) = 0xA0000, core/shim = 0x1D000.
    // 8 words (0x20) per BD. Output path is column 0, rows 0 (shim),
    // 1 (memtile), 2 (core). `mem` is keyed by full tile-encoded addresses,
    // so reconstruct them via TileAddress::encode.
    for (label, col, row, base, is_mem) in [
        ("core", 0u8, 2u8, 0x1D000u32, false),
        ("memtile", 0u8, 1u8, 0xA0000u32, true),
        ("shim", 0u8, 0u8, 0x1D000u32, false),
    ] {
        for bd in 0u32..48 {
            let bd_base = base + bd * 0x20;
            // Word 0 holds buffer length; treat a BD as "present" if any of its
            // 8 words were written.
            let words: Vec<Option<u32>> = (0..8)
                .map(|w| mem.get(&TileAddress::encode(col, row, bd_base + w * 4)).copied())
                .collect();
            if words.iter().all(|w| w.is_none()) {
                continue;
            }
            let ta = TileAddress { col, row, offset: bd_base };
            let g = |w: usize| words[w].unwrap_or(0);

            if is_mem {
                let blen = field(g(0), 16, 0);
                let valid = field(g(7), 31, 31);
                if valid == 0 && blen == 0 {
                    continue;
                }
                let d0_step = field(g(2), 16, 0);
                let d0_wrap = field(g(2), 26, 17);
                let d1_step = field(g(3), 16, 0);
                let d1_wrap = field(g(3), 26, 17);
                let d2_step = field(g(4), 16, 0);
                let d2_wrap = field(g(4), 26, 17);
                let it_step = field(g(6), 16, 0);
                let it_wrap = field(g(6), 22, 17);
                let z0_before = field(g(1), 31, 26);
                let z0_after = field(g(5), 22, 17);
                let z1_after = field(g(5), 27, 23);
                let z2_after = field(g(5), 31, 28);
                println!(
                    "[{label}] tile({},{}) BD{bd} len={blen}w valid={valid}  \
                     d0(step={},wrap={}) d1(step={},wrap={}) d2(step={},wrap={}) it(step={},wrap={})  \
                     zero[d0_before={},d0_after={},d1_after={},d2_after={}]",
                    ta.col,
                    ta.row,
                    d0_step,
                    d0_wrap,
                    d1_step,
                    d1_wrap,
                    d2_step,
                    d2_wrap,
                    it_step,
                    it_wrap,
                    z0_before,
                    z0_after,
                    z1_after,
                    z2_after,
                );
            } else {
                let blen = g(0); // shim length is full 32 bits
                let valid = field(g(7), 25, 25);
                if valid == 0 && blen == 0 {
                    continue;
                }
                let d0_step = field(g(3), 19, 0);
                let d0_wrap = field(g(3), 29, 20);
                let d1_step = field(g(4), 19, 0);
                let d1_wrap = field(g(4), 29, 20);
                let d2_step = field(g(5), 19, 0);
                let it_step = field(g(6), 19, 0);
                let it_wrap = field(g(6), 25, 20);
                println!(
                    "[{label}] tile({},{}) BD{bd} len={blen}w valid={valid}  \
                     d0(step={},wrap={}) d1(step={},wrap={}) d2(step={}) it(step={},wrap={})",
                    ta.col, ta.row, d0_step, d0_wrap, d1_step, d1_wrap, d2_step, it_step, it_wrap,
                );
            }
        }
    }
}
