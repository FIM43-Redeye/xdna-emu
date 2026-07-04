//! Routed memory/MMIO bus: dispatches every firmware load/store to the
//! aperture that owns the address, per spec section 5 (base-0 ROM, data RAM
//! at 0x08b00000, mailbox block at 0x27000000, AIE array windows at
//! 0x04000000, everything else off-array system config).
//!
//! This phase (M1.3): `Rom` and `Ram` are real backing memory; `Mailbox` is a
//! plain-RAM stub (real ring-buffer semantics land with the mailbox protocol
//! work); `Array` and `System` are logged stubs -- routing into `DeviceState`
//! and `sysstub.rs` is later (M2 / M1.6).

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
    /// Everything else (off-array system config); logged stub this phase.
    System,
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
}

impl Bus {
    /// Create a bus over `rom` (the firmware's base-0 `.text`/`.rodata` image).
    /// RAM and mailbox backing stores start empty and grow lazily on first
    /// access, keyed by offset from their region base.
    pub fn new(rom: Vec<u8>) -> Self {
        Self { rom, ram: Vec::new(), mailbox: Vec::new() }
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
        } else {
            Region::System
        }
    }

    /// Read a little-endian 32-bit word.
    pub fn load32(&mut self, addr: u32) -> u32 {
        match Self::region(addr) {
            Region::Rom => read_le32(&self.rom, addr),
            Region::Ram => read_le32(&self.ram, addr - RAM_BASE),
            Region::Mailbox => read_le32(&self.mailbox, addr - MAILBOX_BASE),
            Region::Array => {
                log::debug!("firmware mmio: array load32 stub at 0x{:08X} -> 0", addr);
                0
            }
            Region::System => {
                log::debug!("firmware mmio: system load32 stub at 0x{:08X} -> 0", addr);
                0
            }
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
            Region::Array => {
                log::debug!("firmware mmio: array store32 stub at 0x{:08X} = 0x{:08X}", addr, v);
            }
            Region::System => {
                log::debug!("firmware mmio: system store32 stub at 0x{:08X} = 0x{:08X}", addr, v);
            }
        }
    }

    /// Read a single byte.
    pub fn load8(&mut self, addr: u32) -> u8 {
        match Self::region(addr) {
            Region::Rom => byte_at(&self.rom, addr),
            Region::Ram => byte_at(&self.ram, addr - RAM_BASE),
            Region::Mailbox => byte_at(&self.mailbox, addr - MAILBOX_BASE),
            Region::Array => {
                log::debug!("firmware mmio: array load8 stub at 0x{:08X} -> 0", addr);
                0
            }
            Region::System => {
                log::debug!("firmware mmio: system load8 stub at 0x{:08X} -> 0", addr);
                0
            }
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
            Region::Array => {
                log::debug!("firmware mmio: array store8 stub at 0x{:08X} = 0x{:02X}", addr, v as u8);
            }
            Region::System => {
                log::debug!("firmware mmio: system store8 stub at 0x{:08X} = 0x{:02X}", addr, v as u8);
            }
        }
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
    fn byte_access_is_little_endian_and_independent_of_word_access() {
        let mut bus = Bus::new(vec![]);
        bus.store8(0x08b00200, 0xab);
        bus.store8(0x08b00201, 0xcd);
        assert_eq!(bus.load8(0x08b00200), 0xab);
        assert_eq!(bus.load8(0x08b00201), 0xcd);
        assert_eq!(bus.load32(0x08b00200) & 0xffff, 0xcdab);
    }
}
