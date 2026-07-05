//! Routed memory/MMIO bus: dispatches every firmware load/store to the
//! aperture that owns the address, per spec section 5 (base-0 ROM, data RAM
//! at 0x08b00000, mailbox block at 0x27000000, AIE array windows at
//! 0x04000000, everything else off-array system config).
//!
//! This phase (M1.3 + M1.6): `Rom` and `Ram` are real backing memory;
//! `Mailbox` is a plain-RAM stub (real ring-buffer semantics land with the
//! mailbox protocol work); `Array` is a logged stub (routing into
//! `DeviceState` is M2); `System` is routed through [`crate::firmware::SysStub`],
//! which logs every access and flags waited-on-unmodeled-state spins.

use super::SysStub;

/// The five MMIO apertures a firmware load/store can land in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    /// Base-0 image: `.text`/`.rodata`. Read-only from the firmware's view.
    Rom,
    /// Data RAM window at `0x08b00000` (`.data`/`.bss`).
    Ram,
    /// Mailbox ring/doorbell block at `0x27000000`; plain RAM this phase.
    Mailbox,
    /// AIE array tile/register windows at `0x04000000`; logged stub this phase.
    Array,
    /// Everything else (off-array system config); routed through [`SysStub`].
    System,
    /// Synthesized PSP autorefill page table at `0x3c000000` (M2c); real
    /// physical memory the autorefill walk reads.
    PageTable,
}

/// End of the ROM aperture (exclusive) / start of the array aperture.
const ROM_END: u32 = 0x0400_0000;
/// End of the array aperture (exclusive).
const ARRAY_END: u32 = 0x0800_0000;
/// Start of the RAM aperture.
const RAM_BASE: u32 = 0x08b0_0000;
/// Start of the mailbox aperture.
const MAILBOX_BASE: u32 = 0x2700_0000;
/// End of the mailbox aperture (exclusive).
const MAILBOX_END: u32 = 0x2800_0000;
/// Start of the synthesized page-table aperture.
pub const PAGE_TABLE_BASE: u32 = 0x3c00_0000;
/// End of the synthesized page-table aperture (exclusive). 1 MB window; the
/// code-region PTEs occupy `0x3c080000..` and fit comfortably.
const PAGE_TABLE_END: u32 = 0x3c10_0000;

/// Routed memory/MMIO bus for the Xtensa firmware interpreter.
///
/// Owns the ROM image and the RAM/mailbox backing stores, and routes every
/// access through [`Bus::region`] to the aperture (or stub) that handles it.
pub struct Bus {
    // Base-0 image (`.text`/`.rodata`), sized once at construction.
    rom: Vec<u8>,
    // Data RAM backing store, offset-keyed from `RAM_BASE`, grown lazily.
    ram: Vec<u8>,
    // Mailbox backing store, offset-keyed from `MAILBOX_BASE`, grown lazily.
    mailbox: Vec<u8>,
    // Synthesized page-table backing store, offset-keyed from
    // `PAGE_TABLE_BASE`, grown lazily (M2c).
    page_table: Vec<u8>,
    // Off-array system aperture stub: logs accesses, flags spins.
    sysstub: SysStub,
    // PSP load-offset applied to ROM-region reads: physical `P` reads image
    // byte `P + load_offset`. Zero for `Bus::new`.
    load_offset: u32,
}

impl Bus {
    /// Create a bus over `rom` (the firmware's base-0 `.text`/`.rodata` image).
    /// RAM and mailbox backing stores start empty and grow lazily on first
    /// access, keyed by offset from their region base.
    pub fn new(rom: Vec<u8>) -> Self {
        Self::new_with_load_offset(rom, 0)
    }

    /// Create a bus whose ROM aperture applies the PSP load-offset: a physical
    /// address `P` in the ROM region reads image byte `P + load_offset`
    /// (`phys = file - load_offset`). The x86 PSP loads the firmware body at a
    /// physical base below its file offset; this models that placement so the
    /// code region's virtual->physical map lands on real image bytes (M2c). RAM,
    /// mailbox, array, and system apertures are unaffected.
    pub fn new_with_load_offset(rom: Vec<u8>, load_offset: u32) -> Self {
        Self {
            rom,
            ram: Vec::new(),
            mailbox: Vec::new(),
            page_table: Vec::new(),
            sysstub: SysStub::new(),
            load_offset,
        }
    }

    /// The system-aperture stub, for hang/idle diagnosis (M1.7): its
    /// [`SysStub::spinning`] flags an address the firmware is tight-polling.
    pub fn sysstub(&self) -> &SysStub {
        &self.sysstub
    }

    /// Classify an address into the aperture that owns it, per spec section 5.
    pub fn region(addr: u32) -> Region {
        if addr < ROM_END {
            Region::Rom
        } else if addr < ARRAY_END {
            Region::Array
        } else if (RAM_BASE..MAILBOX_BASE).contains(&addr) {
            Region::Ram
        } else if (MAILBOX_BASE..MAILBOX_END).contains(&addr) {
            Region::Mailbox
        } else if (PAGE_TABLE_BASE..PAGE_TABLE_END).contains(&addr) {
            Region::PageTable
        } else {
            Region::System
        }
    }

    /// Read a little-endian 32-bit word.
    pub fn load32(&mut self, addr: u32) -> u32 {
        match Self::region(addr) {
            Region::Rom => read_le32(&self.rom, addr.wrapping_add(self.load_offset)),
            Region::Ram => read_le32(&self.ram, addr - RAM_BASE),
            Region::Mailbox => read_le32(&self.mailbox, addr - MAILBOX_BASE),
            Region::PageTable => read_le32(&self.page_table, addr - PAGE_TABLE_BASE),
            Region::Array => {
                log::debug!("firmware mmio: array load32 stub at 0x{:08X} -> 0", addr);
                0
            }
            Region::System => self.sysstub.read(addr),
        }
    }

    /// Write a little-endian 32-bit word.
    pub fn store32(&mut self, addr: u32, v: u32) {
        match Self::region(addr) {
            Region::Rom => {
                log::warn!(
                    "firmware mmio: store32 to read-only ROM at 0x{:08X} = 0x{:08X} (ignored)",
                    addr,
                    v
                );
            }
            Region::Ram => write_le32(&mut self.ram, addr - RAM_BASE, v),
            Region::Mailbox => write_le32(&mut self.mailbox, addr - MAILBOX_BASE, v),
            Region::PageTable => write_le32(&mut self.page_table, addr - PAGE_TABLE_BASE, v),
            Region::Array => {
                log::debug!("firmware mmio: array store32 stub at 0x{:08X} = 0x{:08X}", addr, v);
            }
            Region::System => self.sysstub.write(addr, v),
        }
    }

    /// Read a single byte WITHOUT side effects: like [`Bus::load8`] but a
    /// `System`-aperture read returns 0 without logging it or advancing the
    /// [`SysStub`] spin counter. The boot harness uses this to peek the
    /// instruction stream (for call-target symbol tracking) without perturbing
    /// the spin-detection that [`Bus::load8`]'s real fetches drive.
    pub fn peek8(&self, addr: u32) -> u8 {
        match Self::region(addr) {
            Region::Rom => byte_at(&self.rom, addr.wrapping_add(self.load_offset)),
            Region::Ram => byte_at(&self.ram, addr - RAM_BASE),
            Region::Mailbox => byte_at(&self.mailbox, addr - MAILBOX_BASE),
            Region::PageTable => byte_at(&self.page_table, addr - PAGE_TABLE_BASE),
            Region::Array | Region::System => 0,
        }
    }

    /// Read a single byte.
    pub fn load8(&mut self, addr: u32) -> u8 {
        match Self::region(addr) {
            Region::Rom => byte_at(&self.rom, addr.wrapping_add(self.load_offset)),
            Region::Ram => byte_at(&self.ram, addr - RAM_BASE),
            Region::Mailbox => byte_at(&self.mailbox, addr - MAILBOX_BASE),
            Region::PageTable => byte_at(&self.page_table, addr - PAGE_TABLE_BASE),
            Region::Array => {
                log::debug!("firmware mmio: array load8 stub at 0x{:08X} -> 0", addr);
                0
            }
            Region::System => self.sysstub.read(addr) as u8,
        }
    }

    /// Write a single byte (low 8 bits of `v`).
    pub fn store8(&mut self, addr: u32, v: u32) {
        match Self::region(addr) {
            Region::Rom => {
                log::warn!(
                    "firmware mmio: store8 to read-only ROM at 0x{:08X} = 0x{:02X} (ignored)",
                    addr,
                    v as u8
                );
            }
            Region::Ram => set_byte_at(&mut self.ram, addr - RAM_BASE, v as u8),
            Region::Mailbox => set_byte_at(&mut self.mailbox, addr - MAILBOX_BASE, v as u8),
            Region::PageTable => set_byte_at(&mut self.page_table, addr - PAGE_TABLE_BASE, v as u8),
            Region::Array => {
                log::debug!("firmware mmio: array store8 stub at 0x{:08X} = 0x{:02X}", addr, v as u8);
            }
            Region::System => self.sysstub.write(addr, v as u8 as u32),
        }
    }

    /// Populate a word of the synthesized page table (M2c `psp_map`). Physical
    /// address must fall in the PageTable aperture.
    pub fn write_page_table_word(&mut self, phys: u32, v: u32) {
        debug_assert_eq!(Self::region(phys), Region::PageTable);
        write_le32(&mut self.page_table, phys - PAGE_TABLE_BASE, v);
    }

    /// Pre-initialize the data-RAM backing at physical `phys_base` with `data`: a
    /// PSP load segment the firmware expects already resident before it starts (it
    /// never copies it at runtime). Grows the RAM Vec as needed; the region stays
    /// writable (real `.data`/`.bss`). Hardware fact: the x86 PSP places the
    /// relocated `.rodata`/`.data`/`.text`-tail segment at `0x08b00000` before
    /// handing off to the firmware (M2c multi-segment load model).
    pub fn preload_ram(&mut self, phys_base: u32, data: &[u8]) {
        debug_assert_eq!(Self::region(phys_base), Region::Ram, "preload_ram target must be the RAM aperture");
        let off = (phys_base - RAM_BASE) as usize;
        if self.ram.len() < off + data.len() {
            self.ram.resize(off + data.len(), 0);
        }
        self.ram[off..off + data.len()].copy_from_slice(data);
    }
}

/// Read a little-endian 32-bit word from `mem` at `offset`, zero-extending past the end.
fn read_le32(mem: &[u8], offset: u32) -> u32 {
    let o = offset as usize;
    let mut bytes = [0u8; 4];
    for (i, b) in bytes.iter_mut().enumerate() {
        *b = mem.get(o + i).copied().unwrap_or(0);
    }
    u32::from_le_bytes(bytes)
}

/// Write a little-endian 32-bit word into `mem` at `offset`, growing `mem` to fit.
fn write_le32(mem: &mut Vec<u8>, offset: u32, v: u32) {
    let o = offset as usize;
    if mem.len() < o + 4 {
        mem.resize(o + 4, 0);
    }
    mem[o..o + 4].copy_from_slice(&v.to_le_bytes());
}

/// Read a single byte from `mem` at `offset`, zero past the end.
fn byte_at(mem: &[u8], offset: u32) -> u8 {
    mem.get(offset as usize).copied().unwrap_or(0)
}

/// Write a single byte into `mem` at `offset`, growing `mem` to fit.
fn set_byte_at(mem: &mut Vec<u8>, offset: u32, v: u8) {
    let o = offset as usize;
    if mem.len() <= o {
        mem.resize(o + 1, 0);
    }
    mem[o] = v;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routes_addresses_to_regions() {
        assert_eq!(Bus::region(0x00002730), Region::Rom);
        assert_eq!(Bus::region(0x08b00010), Region::Ram);
        assert_eq!(Bus::region(0x27010d00), Region::Mailbox);
        assert_eq!(Bus::region(0x04000000), Region::Array);
        assert_eq!(Bus::region(0xf7000000), Region::System);
    }

    #[test]
    fn rom_reads_little_endian_from_image() {
        let mut bus = Bus::new(vec![0x78, 0x56, 0x34, 0x12]); // @0
        assert_eq!(bus.load32(0), 0x12345678);
    }

    #[test]
    fn ram_round_trips() {
        let mut bus = Bus::new(vec![]);
        bus.store32(0x08b00100, 0xcafebabe);
        assert_eq!(bus.load32(0x08b00100), 0xcafebabe);
    }

    #[test]
    fn mailbox_round_trips_as_ram_this_phase() {
        let mut bus = Bus::new(vec![]);
        bus.store32(0x27010d00, 0x11223344);
        assert_eq!(bus.load32(0x27010d00), 0x11223344);
    }

    #[test]
    fn rom_store_is_logged_and_ignored() {
        let mut bus = Bus::new(vec![0xff; 4]);
        bus.store32(0, 0xdeadbeef);
        // ROM is read-only: the store is a logged violation, not applied.
        assert_eq!(bus.load32(0), 0xffffffff);
    }

    #[test]
    fn array_store_is_stubbed_and_load_returns_zero() {
        let mut bus = Bus::new(vec![]);
        bus.store32(0x04000000, 0x12345678);
        assert_eq!(bus.load32(0x04000000), 0);
    }

    #[test]
    fn system_access_is_stubbed_to_zero() {
        let mut bus = Bus::new(vec![]);
        assert_eq!(bus.load32(0xf7000000), 0);
        bus.store32(0xf7000000, 0xaaaaaaaa); // logged, no effect
        assert_eq!(bus.load32(0xf7000000), 0);
    }

    #[test]
    fn system_access_is_routed_through_sysstub() {
        let mut bus = Bus::new(vec![]);
        bus.load32(0xf7000000);
        bus.load8(0xf7000004);
        bus.store32(0xf7000008, 0x1);
        bus.store8(0xf700000c, 0x2);
        // All four accesses land in the shared SysStub log, visible via the
        // M1.7 diagnostic accessor.
        assert_eq!(bus.sysstub().accesses().len(), 4);
    }

    #[test]
    fn byte_access_is_little_endian_and_independent_of_word_access() {
        let mut bus = Bus::new(vec![]);
        bus.store8(0x08b00200, 0xab);
        bus.store8(0x08b00201, 0xcd);
        assert_eq!(bus.load8(0x08b00200), 0xab);
        assert_eq!(bus.load8(0x08b00201), 0xcd);
        assert_eq!(bus.load32(0x08b00200) & 0xffff, 0xcdab);
    }

    #[test]
    fn rom_access_honors_psp_load_offset() {
        // phys = file - L. With L = 4, physical address 0 reads image byte 4.
        let mut bus = Bus::new_with_load_offset(vec![0, 0, 0, 0, 0x78, 0x56, 0x34, 0x12], 4);
        assert_eq!(bus.load32(0), 0x12345678); // phys 0 -> file 4
        assert_eq!(bus.load8(1), 0x56); // phys 1 -> file 5
                                        // Bus::new keeps offset 0 (regression).
        let mut z = Bus::new(vec![0x78, 0x56, 0x34, 0x12]);
        assert_eq!(z.load32(0), 0x12345678);
    }

    #[test]
    fn preload_ram_initializes_data_region() {
        let mut bus = Bus::new(vec![]);
        // Pre-load 8 bytes at the RAM base + an offset.
        bus.preload_ram(0x08b0_0010, &[0x36, 0xc1, 0x00, 0x4c, 0xde, 0xad, 0xbe, 0xef]);
        assert_eq!(bus.load32(0x08b0_0010), 0x4c00_c136); // little-endian of 36 c1 00 4c
        assert_eq!(bus.load32(0x08b0_0014), 0xefbe_adde);
        // Unwritten RAM stays zero; region routing unaffected.
        assert_eq!(bus.load32(0x08b0_0000), 0);
        assert_eq!(Bus::region(0x08b0_0010), Region::Ram);
        // Pre-loaded RAM is still writable (it's .data/.bss, not ROM).
        bus.store32(0x08b0_0010, 0x1234_5678);
        assert_eq!(bus.load32(0x08b0_0010), 0x1234_5678);
    }

    #[test]
    fn page_table_aperture_round_trips() {
        let mut bus = Bus::new(vec![]);
        assert_eq!(Bus::region(0x3c08_0000), Region::PageTable);
        bus.write_page_table_word(0x3c08_0000, 0x08b0_5001);
        assert_eq!(bus.load32(0x3c08_0000), 0x08b0_5001);
        // Below and above the aperture is still System (regression).
        assert_eq!(Bus::region(0x3c10_0000), Region::System);
    }
}
