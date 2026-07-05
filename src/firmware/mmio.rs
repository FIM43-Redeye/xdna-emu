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

/// One recorded stub-aperture access (Array / Mailbox / System), captured when
/// the [`Bus`] probe is armed. Diagnostic instrument for the M2c Phase 2 boot
/// walk: a peripheral read that returns a stub value (0) which the firmware then
/// branches on is the suspected source of a wrong boot path. `pc` is the CPU PC
/// at the access (threaded in by the boot harness via [`Bus::set_probe_pc`]);
/// `seq` is the monotonic access index within the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StubAccess {
    /// CPU PC at the moment of the access (harness-supplied; 0 if unset).
    pub pc: u32,
    /// Physical address accessed (post-translation).
    pub addr: u32,
    /// Which stub aperture the address fell in.
    pub region: Region,
    /// Value read, or value written.
    pub value: u32,
    /// Access width in bytes (1 or 4).
    pub width: u8,
    /// True for a store, false for a load.
    pub is_write: bool,
    /// Monotonic index of this access within the armed run.
    pub seq: u64,
}

/// End of the ROM aperture (exclusive) / start of the array aperture.
const ROM_END: u32 = 0x0400_0000;
/// End of the array aperture (exclusive).
const ARRAY_END: u32 = 0x0800_0000;
/// Start of the RAM aperture.
const RAM_BASE: u32 = 0x08b0_0000;
/// Start of the mailbox aperture.
pub const MAILBOX_BASE: u32 = 0x2700_0000;
/// End of the mailbox aperture (exclusive).
pub const MAILBOX_END: u32 = 0x2800_0000;
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
    // Diagnostic stub-access probe (M2c Phase 2 boot-walk instrument). `None`
    // by default -- zero cost when disarmed. When `Some`, every Array/Mailbox/
    // System access appends a `StubAccess` tagged with `probe_pc`.
    probe: Option<Vec<StubAccess>>,
    // The PC the boot harness last set; stamped onto recorded accesses.
    probe_pc: u32,
    // Monotonic access counter for the armed run (`StubAccess::seq`).
    probe_seq: u64,
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
            probe: None,
            probe_pc: 0,
            probe_seq: 0,
        }
    }

    /// Arm the diagnostic stub-access probe: from now on, every Array / Mailbox /
    /// System load or store is recorded (with the last [`Bus::set_probe_pc`] PC)
    /// until [`Bus::take_probe`] drains it. No effect on production paths when
    /// left disarmed. M2c Phase 2 boot-walk instrument.
    pub fn arm_probe(&mut self) {
        self.probe = Some(Vec::new());
        self.probe_seq = 0;
    }

    /// Set the PC stamped onto subsequently recorded stub accesses. The boot
    /// harness calls this once per instruction, before stepping the CPU.
    pub fn set_probe_pc(&mut self, pc: u32) {
        self.probe_pc = pc;
    }

    /// Drain the recorded stub-access log, disarming the probe.
    pub fn take_probe(&mut self) -> Vec<StubAccess> {
        self.probe.take().unwrap_or_default()
    }

    /// Record a stub-aperture access if the probe is armed. Called from the
    /// load/store paths for Array / Mailbox / System addresses.
    fn record_stub(&mut self, addr: u32, region: Region, value: u32, width: u8, is_write: bool) {
        if let Some(log) = self.probe.as_mut() {
            log.push(StubAccess {
                pc: self.probe_pc,
                addr,
                region,
                value,
                width,
                is_write,
                seq: self.probe_seq,
            });
            self.probe_seq += 1;
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
            Region::Mailbox => {
                let v = read_le32(&self.mailbox, addr - MAILBOX_BASE);
                self.record_stub(addr, Region::Mailbox, v, 4, false);
                v
            }
            Region::PageTable => read_le32(&self.page_table, addr - PAGE_TABLE_BASE),
            Region::Array => {
                log::debug!("firmware mmio: array load32 stub at 0x{:08X} -> 0", addr);
                self.record_stub(addr, Region::Array, 0, 4, false);
                0
            }
            Region::System => {
                let v = self.sysstub.read(addr);
                self.record_stub(addr, Region::System, v, 4, false);
                v
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
            Region::Mailbox => {
                write_le32(&mut self.mailbox, addr - MAILBOX_BASE, v);
                self.record_stub(addr, Region::Mailbox, v, 4, true);
            }
            Region::PageTable => write_le32(&mut self.page_table, addr - PAGE_TABLE_BASE, v),
            Region::Array => {
                log::debug!("firmware mmio: array store32 stub at 0x{:08X} = 0x{:08X}", addr, v);
                self.record_stub(addr, Region::Array, v, 4, true);
            }
            Region::System => {
                self.sysstub.write(addr, v);
                self.record_stub(addr, Region::System, v, 4, true);
            }
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
                self.record_stub(addr, Region::Array, 0, 1, false);
                0
            }
            Region::System => {
                let v = self.sysstub.read(addr);
                self.record_stub(addr, Region::System, v, 1, false);
                v as u8
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
            Region::Mailbox => {
                set_byte_at(&mut self.mailbox, addr - MAILBOX_BASE, v as u8);
                self.record_stub(addr, Region::Mailbox, v as u8 as u32, 1, true);
            }
            Region::PageTable => set_byte_at(&mut self.page_table, addr - PAGE_TABLE_BASE, v as u8),
            Region::Array => {
                log::debug!("firmware mmio: array store8 stub at 0x{:08X} = 0x{:02X}", addr, v as u8);
                self.record_stub(addr, Region::Array, v as u8 as u32, 1, true);
            }
            Region::System => {
                self.sysstub.write(addr, v as u8 as u32);
                self.record_stub(addr, Region::System, v as u8 as u32, 1, true);
            }
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

    /// Region-aware bulk fill: write `pattern` (1, 2, or 4 bytes, little-endian
    /// store order) repeated to cover `byte_len` bytes starting at physical
    /// `phys`. Semantically identical to `byte_len / pattern.len()` successive
    /// `store8`/`store16`/`store32`s at consecutive addresses: Rom/Array/System
    /// drop the write (matching their per-store stubs), Ram/Mailbox/PageTable
    /// fill their backing store. The interpreter's fill-loop fast-path uses this
    /// to collapse a large memset without grinding it byte-by-byte. Caller
    /// guarantees the whole `[phys, phys+byte_len)` range stays within one
    /// region (it chunks per page and re-resolves the region for each chunk),
    /// `pattern.len()` is 1/2/4, and `byte_len` is a multiple of `pattern.len()`
    /// with the chunk starting at pattern phase 0.
    pub fn fill_pattern(&mut self, phys: u32, pattern: &[u8], byte_len: usize) {
        debug_assert!(matches!(pattern.len(), 1 | 2 | 4));
        debug_assert_eq!(byte_len % pattern.len(), 0);
        match Self::region(phys) {
            // These apertures drop stores (read-only image / logged stub); a
            // bulk fill is the same no-op as the per-store path.
            Region::Rom | Region::Array | Region::System => {}
            Region::Ram => fill_mem(&mut self.ram, phys - RAM_BASE, pattern, byte_len),
            Region::Mailbox => fill_mem(&mut self.mailbox, phys - MAILBOX_BASE, pattern, byte_len),
            Region::PageTable => fill_mem(&mut self.page_table, phys - PAGE_TABLE_BASE, pattern, byte_len),
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

/// Fill `mem[offset .. offset+byte_len]` with `pattern` repeated (phase 0),
/// growing `mem` to fit. `pattern.len()` is 1/2/4.
fn fill_mem(mem: &mut Vec<u8>, offset: u32, pattern: &[u8], byte_len: usize) {
    let o = offset as usize;
    if mem.len() < o + byte_len {
        mem.resize(o + byte_len, 0);
    }
    let dst = &mut mem[o..o + byte_len];
    match pattern.len() {
        1 => dst.fill(pattern[0]),
        w => {
            for (i, b) in dst.iter_mut().enumerate() {
                *b = pattern[i % w];
            }
        }
    }
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
    fn fill_pattern_matches_repeated_stores() {
        let mut bus = Bus::new(vec![]);
        // Byte fill into RAM.
        bus.fill_pattern(0x08b0_1000, &[0xab], 10);
        for a in 0x08b0_1000..0x08b0_100a {
            assert_eq!(bus.load8(a), 0xab, "byte fill @ {a:#x}");
        }
        assert_eq!(bus.load8(0x08b0_100a), 0, "one past the fill is untouched");
        // Word fill into RAM: 0xdeadbeef repeated, little-endian.
        bus.fill_pattern(0x08b0_2000, &0xdead_beefu32.to_le_bytes(), 8);
        assert_eq!(bus.load32(0x08b0_2000), 0xdead_beef);
        assert_eq!(bus.load32(0x08b0_2004), 0xdead_beef);
        // Rom/System fills are dropped (no panic, no effect).
        bus.fill_pattern(0x0000_1000, &[0xff], 0x1000); // Rom: dropped
        assert_eq!(bus.load8(0x0000_1000), bus.load8(0x0000_1000)); // no crash
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
