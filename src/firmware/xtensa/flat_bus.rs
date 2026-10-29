//! A plain flat memory implementing [`XtensaBus`]: one little-endian byte
//! space shared by instruction fetch, literals and data (no Harvard split, no
//! apertures, no views). Backing is sparse 4 KiB pages allocated on first
//! write, so tests may touch widely separated addresses cheaply. Unwritten
//! bytes read as zero.

use std::collections::HashMap;

use super::bus::XtensaBus;

const PAGE_BITS: u32 = 12;
const PAGE_SIZE: usize = 1 << PAGE_BITS;

pub struct FlatBus {
    pages: HashMap<u32, Box<[u8; PAGE_SIZE]>>,
}

impl FlatBus {
    /// A bus whose bytes `0..rom.len()` start as `rom`.
    pub fn new(rom: Vec<u8>) -> Self {
        let mut bus = Self { pages: HashMap::new() };
        for (i, b) in rom.into_iter().enumerate() {
            if b != 0 {
                bus.write8(i as u32, b);
            }
        }
        bus
    }

    pub fn read8(&self, addr: u32) -> u8 {
        self.pages
            .get(&(addr >> PAGE_BITS))
            .map_or(0, |p| p[addr as usize & (PAGE_SIZE - 1)])
    }

    pub fn write8(&mut self, addr: u32, v: u8) {
        let page = self
            .pages
            .entry(addr >> PAGE_BITS)
            .or_insert_with(|| Box::new([0u8; PAGE_SIZE]));
        page[addr as usize & (PAGE_SIZE - 1)] = v;
    }

    pub fn read32(&self, addr: u32) -> u32 {
        u32::from_le_bytes([
            self.read8(addr),
            self.read8(addr.wrapping_add(1)),
            self.read8(addr.wrapping_add(2)),
            self.read8(addr.wrapping_add(3)),
        ])
    }

    pub fn write32(&mut self, addr: u32, v: u32) {
        for (i, b) in v.to_le_bytes().into_iter().enumerate() {
            self.write8(addr.wrapping_add(i as u32), b);
        }
    }
}

impl XtensaBus for FlatBus {
    fn fetch8(&mut self, _vaddr: u32, paddr: u32) -> u8 {
        self.read8(paddr)
    }
    fn inst_load8(&mut self, paddr: u32) -> u8 {
        self.read8(paddr)
    }
    fn literal_load32(&mut self, _vaddr: u32, paddr: u32) -> u32 {
        self.read32(paddr)
    }
    fn data_load32(&mut self, _pc: u32, paddr: u32) -> u32 {
        self.read32(paddr)
    }
    fn data_load8(&mut self, paddr: u32) -> u8 {
        self.read8(paddr)
    }
    fn data_store32(&mut self, paddr: u32, v: u32) {
        self.write32(paddr, v);
    }
    fn data_store8(&mut self, paddr: u32, v: u32) {
        self.write8(paddr, v as u8);
    }
    fn pte_load32(&mut self, paddr: u32) -> u32 {
        self.read32(paddr)
    }
}
