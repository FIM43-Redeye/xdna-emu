//! Routed memory/MMIO bus: dispatches every firmware load/store to the
//! aperture that owns the address, per spec section 5 (base-0 ROM, data RAM
//! at 0x08b00000, mailbox block at 0x27000000, AIE array windows at
//! 0x04000000, everything else off-array system config).
//!
//! `Rom` and `Ram` are real backing memory; `Mailbox` is RAM-backed except for
//! derived controller-register behavior; `Array` routes 32-bit accesses into a
//! transiently borrowed [`DeviceState`] while the firmware and array
//! interpreters run together, and otherwise falls back to a logged stub;
//! `System` is routed through [`crate::firmware::SysStub`], which logs every
//! access and flags waited-on-unmodeled-state spins.

use super::SysStub;
use crate::device::DeviceState;

/// The five MMIO apertures a firmware load/store can land in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    /// Base-0 image: `.text`/`.rodata`. Read-only from the firmware's view.
    Rom,
    /// Data RAM window at `0x08b00000` (`.data`/`.bss`).
    Ram,
    /// Mailbox ring/doorbell block at `0x27000000`; mostly RAM-backed this phase.
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
/// End (exclusive) of the low virtual window that maps to local memory. A DATA
/// access below this vaddr goes to the Harvard local data memory (`local_data`),
/// not the image; an instruction fetch below it still reads the image (local
/// IRAM). Coincides numerically with `ROM_END`, but is a VIRTUAL-address
/// predicate applied before translation. See the M2c Harvard-model spec.
pub const LOCAL_DATA_END: u32 = 0x0400_0000;
/// Phoenix BAR0 registers, BAR2 SRAM, and BAR4 mailbox form one contiguous
/// device aperture. The open NPU1 driver supplies the three bases and live PCI
/// resources supply their sizes.
const PHOENIX_DEVICE_BASE: u32 = 0x0300_0000;
const PHOENIX_DEVICE_END: u32 = 0x0310_0000;
/// End of the array aperture (exclusive).
const ARRAY_END: u32 = 0x0800_0000;
/// Firmware array base: the AIE2 tile aperture starts here (coincides with
/// `ROM_END`). Tiles are addressed `ARRAY_BASE + col<<TILE_COL_SHIFT +
/// row<<TILE_ROW_SHIFT + reg`.
/// ponytail: aperture only spans cols 0..=1 (col 2 lands at ARRAY_END); real
/// multi-column jobs need the true firmware->column map (open M2/M3 question --
/// the col<<25 base collides with RAM at col>=5, so the firmware is not simply
/// col-absolute here). Enough for the M1 seam + its cols-0/1 unit test.
const ARRAY_BASE: u32 = ROM_END;
/// Start of the RAM aperture.
const RAM_BASE: u32 = 0x08b0_0000;
/// Start of the mailbox aperture.
pub const MAILBOX_BASE: u32 = 0x2700_0000;
/// End of the mailbox aperture (exclusive).
pub const MAILBOX_END: u32 = 0x2800_0000;
/// Four interrupt pending/ack banks. Firmware helpers at `0x899c..0x89d4`
/// write one-hot acknowledgements here and persistent enables at `0x27200300`.
const MAILBOX_IRQ_ACK_BASE: u32 = 0x2720_03b0;
const MAILBOX_IRQ_ACK_END: u32 = 0x2720_03c0;
/// Phoenix exposes 16 local-SRAM aliases in each direction as interleaved
/// 4 KiB device windows. The open driver pins X2I slots 0/1 and I2X slot 15;
/// the firmware's alias helper supplies the matching config-register formula.
const HOST_SRAM_ALIAS_BASE: u32 = 0x030a_0000;
const HOST_SRAM_ALIAS_WINDOW_SIZE: u32 = 0x1000;
const HOST_SRAM_ALIAS_COUNT: u32 = 32;
const SRAM_ALIAS_CONFIG_BASE: u32 = 0x2721_0084;
const SRAM_ALIAS_DIRECTION_STRIDE: u32 = 0x40;
const SRAM_ALIAS_LOCAL_MASK: u32 = 0x0007_ffff;
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
    // Local data memory (Xtensa DRAM): a writable, zero-initialized backing for
    // low-window data accesses (vaddr < LOCAL_DATA_END), offset-keyed from 0
    // and grown lazily on initialization or write. Physically distinct from
    // `rom` (the image / local IRAM): the firmware's boot memset zeroes this
    // memory, not its own code.
    local_data: Vec<u8>,
    // Off-array system aperture stub: logs accesses, flags spins.
    sysstub: SysStub,
    // PSP load-offset applied to ROM-region reads: physical `P` reads image
    // byte `P + load_offset`. Zero for `Bus::new`.
    load_offset: u32,
    // Piecewise ROM fetch file-offset overrides: `(vaddr_lo, vaddr_hi, file_offset)`.
    // The firmware .text is NOT a single uniform file offset -- the PSP places
    // some sections at a link (VMA) address that does not follow the file (LMA)
    // linearly. A low-window FETCH whose vaddr is in `[vaddr_lo, vaddr_hi)` uses
    // `file_offset` instead of `load_offset`. Keyed on vaddr (see `fetch8`), not
    // phys, so the code region's alias of the same phys is unaffected. M2c iter16.
    rom_overlays: Vec<(u32, u32, u32)>,
    // Diagnostic stub-access probe (M2c Phase 2 boot-walk instrument). `None`
    // by default -- zero cost when disarmed. When `Some`, every Array/Mailbox/
    // System access appends a `StubAccess` tagged with `probe_pc`.
    probe: Option<Vec<StubAccess>>,
    // The PC the boot harness last set; stamped onto recorded accesses.
    probe_pc: u32,
    // Monotonic access counter for the armed run (`StubAccess::seq`).
    probe_seq: u64,
}

/// CPU-facing bus view. Standalone firmware keeps array MMIO stubbed; an
/// integrated step borrows the array interpreter's sole `DeviceState`.
pub(crate) enum CpuBus<'a> {
    Standalone(&'a mut Bus),
    WithDevice {
        bus: &'a mut Bus,
        device: &'a mut DeviceState,
    },
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
            local_data: Vec::new(),
            sysstub: SysStub::new(),
            load_offset,
            rom_overlays: Vec::new(),
            probe: None,
            probe_pc: 0,
            probe_seq: 0,
        }
    }

    /// Borrow `device` for a firmware step. The array interpreter remains the
    /// sole owner; this view only routes synchronous firmware array MMIO.
    pub(crate) fn with_device<'a>(&'a mut self, device: &'a mut DeviceState) -> CpuBus<'a> {
        CpuBus::WithDevice { bus: self, device }
    }

    /// Register a piecewise ROM file-offset override for FETCHES in the low
    /// window: a fetch whose VIRTUAL address falls in `[vaddr_lo, vaddr_hi)`
    /// reads image byte `vaddr + file_offset` instead of `vaddr + load_offset`.
    /// Models a firmware section the PSP places at a link (VMA) address that does
    /// not follow the file (LMA) linearly (M2c iter16: the low dispatch-function
    /// block at file = vaddr + 0x100).
    ///
    /// Keyed on the fetch VADDR, not the physical address, on purpose: the code
    /// region (`0x2000_0000+`) maps to the SAME low physical range and must keep
    /// the base offset -- the two only diverge in virtual space. This is the same
    /// code-region/low-window collision that forces [`Bus::is_local_data`] to be a
    /// vaddr predicate.
    pub fn add_rom_overlay(&mut self, vaddr_lo: u32, vaddr_hi: u32, file_offset: u32) {
        self.rom_overlays.push((vaddr_lo, vaddr_hi, file_offset));
    }

    #[cfg(test)]
    pub(crate) fn remove_rom_overlay(&mut self, vaddr_lo: u32, vaddr_hi: u32) {
        self.rom_overlays.retain(|&(lo, hi, _)| (lo, hi) != (vaddr_lo, vaddr_hi));
    }

    /// Fetch one instruction byte at virtual address `vaddr` (already translated
    /// to `phys`). A low-window fetch inside a registered overlay reads the
    /// overlay's file bytes; every other fetch -- including code-region aliases of
    /// the same physical address -- uses the normal physical path.
    pub fn fetch8(&mut self, vaddr: u32, phys: u32) -> u8 {
        for &(lo, hi, off) in &self.rom_overlays {
            if (lo..hi).contains(&vaddr) {
                return byte_at(&self.rom, vaddr.wrapping_add(off));
            }
        }
        self.inst_load8(phys)
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

    /// Scan every backing store for a byte pattern, returning `(region, addr)`
    /// for each occurrence (`addr` is the firmware-local address = region base +
    /// offset). Mechanism-independent evidence tool: a struct that was copied to
    /// host-visible memory shows its magic bytes at a new address regardless of
    /// the store width used. `rom` carries the instruction image; `local_data`,
    /// `mailbox`, `ram`, and `page_table` are runtime regions.
    #[cfg(test)]
    pub(crate) fn scan_bytes(&self, needle: &[u8]) -> Vec<(&'static str, u32)> {
        let find = |name: &'static str, buf: &[u8], base: u32| -> Vec<(&'static str, u32)> {
            if needle.is_empty() || buf.len() < needle.len() {
                return Vec::new();
            }
            (0..=buf.len() - needle.len())
                .filter(|&i| &buf[i..i + needle.len()] == needle)
                .map(|i| (name, base.wrapping_add(i as u32)))
                .collect()
        };
        let mut hits = Vec::new();
        hits.extend(find("rom", &self.rom, 0));
        hits.extend(find("local_data", &self.local_data, 0));
        hits.extend(find("ram", &self.ram, RAM_BASE));
        hits.extend(find("mailbox", &self.mailbox, MAILBOX_BASE));
        hits.extend(find("page_table", &self.page_table, PAGE_TABLE_BASE));
        hits
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

    /// Decode a firmware Array-aperture physical address into `(col, row,
    /// offset)`. The firmware's array window uses `tile(col,row,reg) =
    /// ARRAY_BASE + col<<TILE_COL_SHIFT + row<<TILE_ROW_SHIFT + reg` -- the same
    /// AIE2 tile geometry the device model uses. Shifts are derived from the
    /// archspec; the base is the firmware aperture (NOT the runtime-sequence
    /// 0x200_0000_0000 encoding that `decode_npu_address` handles, and with no
    /// start_col relocation -- the firmware addresses physical tiles directly).
    pub fn decode_array_addr(addr: u32) -> (u8, u8, u32) {
        use xdna_archspec::aie2::{TILE_COL_SHIFT, TILE_OFFSET_MASK, TILE_ROW_SHIFT};
        let row_mask = (1u32 << (TILE_COL_SHIFT - TILE_ROW_SHIFT)) - 1;
        let rel = addr.wrapping_sub(ARRAY_BASE);
        let col = (rel >> TILE_COL_SHIFT) as u8;
        let row = ((rel >> TILE_ROW_SHIFT) & row_mask) as u8;
        let offset = rel & TILE_OFFSET_MASK;
        (col, row, offset)
    }

    /// Classify an address into the aperture that owns it, per spec section 5.
    pub fn region(addr: u32) -> Region {
        if Self::is_phoenix_device(addr) {
            Region::System
        } else if addr < ROM_END {
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

    /// True iff `vaddr` is a low-window virtual address whose DATA accesses go
    /// to local memory (`local_data`). A vaddr predicate, applied before
    /// translation -- the local/image split cannot be made on the physical
    /// address, because the code region and the low window collide there.
    pub fn is_local_data(vaddr: u32) -> bool {
        vaddr < LOCAL_DATA_END && !Self::is_phoenix_device(vaddr)
    }

    fn is_phoenix_device(addr: u32) -> bool {
        (PHOENIX_DEVICE_BASE..PHOENIX_DEVICE_END).contains(&addr)
    }

    /// Read a little-endian 32-bit word from local data memory at `off` (== the
    /// low-window vaddr). Blank (0) past the written extent.
    pub fn load_local32(&self, off: u32) -> u32 {
        read_le32(&self.local_data, off)
    }

    /// Read a byte from local data memory at `off`.
    pub fn load_local8(&self, off: u32) -> u8 {
        byte_at(&self.local_data, off)
    }

    /// Write a little-endian 32-bit word to local data memory at `off`, growing
    /// the backing to fit.
    pub fn store_local32(&mut self, off: u32, v: u32) {
        write_le32(&mut self.local_data, off, v);
    }

    /// Write the low byte of `v` to local data memory at `off`, growing to fit.
    pub fn store_local8(&mut self, off: u32, v: u32) {
        set_byte_at(&mut self.local_data, off, v as u8);
    }

    fn host_sram_local_offset(&self, addr: u32, width: u32) -> Option<u32> {
        let rel = addr.checked_sub(HOST_SRAM_ALIAS_BASE)?;
        let alias_index = rel / HOST_SRAM_ALIAS_WINDOW_SIZE;
        if alias_index >= HOST_SRAM_ALIAS_COUNT {
            return None;
        }

        let window_offset = rel % HOST_SRAM_ALIAS_WINDOW_SIZE;
        let slot = alias_index / 2;
        let direction = alias_index % 2;
        let config_addr = SRAM_ALIAS_CONFIG_BASE + slot * 4 + direction * SRAM_ALIAS_DIRECTION_STRIDE;
        let config = read_le32(&self.mailbox, config_addr - MAILBOX_BASE);
        let size = (config >> 19) + 1;
        (window_offset.checked_add(width)? <= size)
            .then_some((config & SRAM_ALIAS_LOCAL_MASK) + window_offset)
    }

    /// Read a host-visible Phoenix SRAM alias. Alias geometry comes from the
    /// firmware-programmed config registers; an unmapped access reads zero.
    pub fn host_sram_load32(&self, addr: u32) -> u32 {
        self.host_sram_local_offset(addr, 4)
            .map_or(0, |off| read_le32(&self.local_data, off))
    }

    /// Write a host-visible Phoenix SRAM alias. The open driver uses this path
    /// to clear `FW_ALIVE_OFF`; an unmapped access is dropped.
    pub fn host_sram_store32(&mut self, addr: u32, v: u32) {
        if let Some(off) = self.host_sram_local_offset(addr, 4) {
            write_le32(&mut self.local_data, off, v);
        }
    }

    /// Supply an I2X alias established before the management CPU starts.
    pub(super) fn preconfigure_i2x_sram_alias(&mut self, slot: u32, local_base: u32, size: u32) {
        debug_assert!(slot < HOST_SRAM_ALIAS_COUNT / 2);
        debug_assert!(size.is_power_of_two() && size <= HOST_SRAM_ALIAS_WINDOW_SIZE);
        debug_assert_eq!(local_base & (size - 1), 0);
        debug_assert!(local_base + size <= SRAM_ALIAS_LOCAL_MASK + 1);

        let config_addr = SRAM_ALIAS_CONFIG_BASE + slot * 4 + SRAM_ALIAS_DIRECTION_STRIDE;
        let config = ((size - 1) << 19) | local_base;
        write_le32(&mut self.mailbox, config_addr - MAILBOX_BASE, config);
    }

    /// Bulk fill of local data memory: `pattern` (1/2/4 bytes, little-endian
    /// store order) repeated to cover `byte_len` bytes at `off`. Byte-identical
    /// to that many `store_local8`/`16`/`32`s. Zero-pattern optimization: a
    /// zero fill never GROWS the backing (unwritten offsets already read 0); it
    /// only clears the already-populated prefix.
    pub fn fill_local(&mut self, off: u32, pattern: &[u8], byte_len: usize) {
        debug_assert!(matches!(pattern.len(), 1 | 2 | 4));
        debug_assert_eq!(byte_len % pattern.len(), 0);
        if pattern.iter().all(|&b| b == 0) {
            let o = off as usize;
            let end = o + byte_len;
            let cap = end.min(self.local_data.len());
            if cap > o {
                self.local_data[o..cap].fill(0);
            }
        } else {
            fill_mem(&mut self.local_data, off, pattern, byte_len);
        }
    }

    /// Test-only: current length of the local-data backing (to assert the
    /// zero-fill allocation cap does not grow it).
    #[cfg(test)]
    pub fn local_data_len_for_test(&self) -> usize {
        self.local_data.len()
    }

    /// Region-dispatch read of a little-endian 32-bit word by PHYSICAL
    /// address. Private: side-explicit callers ([`Bus::data_load32`],
    /// [`Bus::inst_load32`]) intercept the low (`local_data`/`rom`) window
    /// themselves and call this only for the high span, where I-side and
    /// D-side share the same aperture behavior. Not exposed directly --
    /// an ambiguous bare accessor can't tell which Harvard side a caller meant.
    fn region_load32(&mut self, addr: u32, device: Option<&mut DeviceState>) -> u32 {
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
                let v = if let Some(dev) = device {
                    let (col, row, offset) = Self::decode_array_addr(addr);
                    dev.read_tile_register(col, row, offset)
                } else {
                    log::debug!("firmware mmio: array load32 stub at 0x{:08X} -> 0", addr);
                    0
                };
                self.record_stub(addr, Region::Array, v, 4, false);
                v
            }
            Region::System => {
                let v = self.sysstub.read(addr);
                self.record_stub(addr, Region::System, v, 4, false);
                v
            }
        }
    }

    /// Region-dispatch write of a little-endian 32-bit word by PHYSICAL
    /// address. Private -- see [`Bus::region_load32`]; [`Bus::data_store32`]
    /// intercepts the low window and calls this only for the high span.
    fn region_store32(&mut self, addr: u32, v: u32, device: Option<&mut DeviceState>) {
        match Self::region(addr) {
            // Unreachable via the public API: every Rom-region paddr is < LOCAL_DATA_END
            // and intercepted by data_store32 before region_store32 is reached. Kept for
            // match exhaustiveness.
            Region::Rom => {
                log::warn!(
                    "firmware mmio: store32 to read-only ROM at 0x{:08X} = 0x{:08X} (ignored)",
                    addr,
                    v
                );
            }
            Region::Ram => write_le32(&mut self.ram, addr - RAM_BASE, v),
            Region::Mailbox => {
                let off = addr - MAILBOX_BASE;
                if (MAILBOX_IRQ_ACK_BASE..MAILBOX_IRQ_ACK_END).contains(&addr) {
                    let pending = read_le32(&self.mailbox, off);
                    write_le32(&mut self.mailbox, off, pending & !v);
                } else {
                    write_le32(&mut self.mailbox, off, v);
                }
                self.record_stub(addr, Region::Mailbox, v, 4, true);
            }
            Region::PageTable => write_le32(&mut self.page_table, addr - PAGE_TABLE_BASE, v),
            Region::Array => {
                if let Some(dev) = device {
                    let (col, row, offset) = Self::decode_array_addr(addr);
                    dev.write_tile_register(col, row, offset, v);
                } else {
                    log::debug!("firmware mmio: array store32 stub at 0x{:08X} = 0x{:08X}", addr, v);
                }
                self.record_stub(addr, Region::Array, v, 4, true);
            }
            Region::System => {
                self.sysstub.write(addr, v);
                self.record_stub(addr, Region::System, v, 4, true);
            }
        }
    }

    /// Read a single byte WITHOUT side effects: like [`Bus::region_load8`] but
    /// a `System`-aperture read returns 0 without logging it or advancing the
    /// [`SysStub`] spin counter. The boot harness uses this to peek the
    /// instruction stream (for call-target symbol tracking) without perturbing
    /// the spin-detection that a real fetch drives.
    pub fn peek8(&self, addr: u32) -> u8 {
        match Self::region(addr) {
            Region::Rom => byte_at(&self.rom, addr.wrapping_add(self.load_offset)),
            Region::Ram => byte_at(&self.ram, addr - RAM_BASE),
            Region::Mailbox => byte_at(&self.mailbox, addr - MAILBOX_BASE),
            Region::PageTable => byte_at(&self.page_table, addr - PAGE_TABLE_BASE),
            Region::Array | Region::System => 0,
        }
    }

    /// Region-dispatch read of a single byte by PHYSICAL address. Private --
    /// see [`Bus::region_load32`]; [`Bus::data_load8`]/[`Bus::inst_load8`]
    /// intercept the low window and call this only for the high span.
    fn region_load8(&mut self, addr: u32) -> u8 {
        match Self::region(addr) {
            Region::Rom => byte_at(&self.rom, addr.wrapping_add(self.load_offset)),
            Region::Ram => byte_at(&self.ram, addr - RAM_BASE),
            Region::Mailbox => byte_at(&self.mailbox, addr - MAILBOX_BASE),
            Region::PageTable => byte_at(&self.page_table, addr - PAGE_TABLE_BASE),
            Region::Array => {
                // ponytail: 8-bit array access left stubbed -- AIE tile registers
                // are 32-bit MMIO; the firmware programs them word-wise. Wire a
                // byte path only if a real byte access to the array shows up.
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

    /// Region-dispatch write of a single byte (low 8 bits of `v`) by PHYSICAL
    /// address. Private -- see [`Bus::region_load32`]; [`Bus::data_store8`]
    /// intercepts the low window and calls this only for the high span.
    fn region_store8(&mut self, addr: u32, v: u32) {
        match Self::region(addr) {
            // Unreachable via the public API: every Rom-region paddr is < LOCAL_DATA_END
            // and intercepted by data_store8 before region_store8 is reached. Kept for
            // match exhaustiveness.
            Region::Rom => {
                log::warn!(
                    "firmware mmio: store8 to read-only ROM at 0x{:08X} = 0x{:02X} (ignored)",
                    addr,
                    v as u8
                );
            }
            Region::Ram => set_byte_at(&mut self.ram, addr - RAM_BASE, v as u8),
            Region::Mailbox => {
                let off = addr - MAILBOX_BASE;
                if (MAILBOX_IRQ_ACK_BASE..MAILBOX_IRQ_ACK_END).contains(&addr) {
                    let pending = byte_at(&self.mailbox, off);
                    set_byte_at(&mut self.mailbox, off, pending & !(v as u8));
                } else {
                    set_byte_at(&mut self.mailbox, off, v as u8);
                }
                self.record_stub(addr, Region::Mailbox, v as u8 as u32, 1, true);
            }
            Region::PageTable => set_byte_at(&mut self.page_table, addr - PAGE_TABLE_BASE, v as u8),
            Region::Array => {
                // ponytail: see region_load8 -- 8-bit array access stays stubbed.
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

    /// Stand in for the PSP: fill the synthesized page table with IDENTITY PTEs
    /// (paddr == vaddr) over `[vaddr_lo, vaddr_hi)`, with attribute nibble `attr`
    /// and ring 0. The real firmware configures page-table mode (`PTEVADDR`,
    /// `DTLBCFG`) and invalidates its coarse identity mappings, then relies on a
    /// page table the PSP has already populated in DRAM -- it never writes the PT
    /// itself (0 stores over a full boot; see the boot-wake finding's iter24
    /// discriminator). We own the emulator's physical layout, so we supply a
    /// functionally-equivalent table: identity is the natural, self-consistent
    /// choice for these apertures, and `attr` carries the firmware's own declared
    /// intent (its transient way-5 bootstrap install used attr 7 = RWX; the reset
    /// identity entries use attr 3). The PTE format is `(paddr & mask) | ring<<4 |
    /// attr`, the same layout `Mmu::decode_pte` reads on autorefill. Entries
    /// outside a firmware-configured PT aperture are never consulted (autorefill
    /// only fires on a resident-TLB miss), so this only affects addresses the
    /// firmware genuinely delegates to the PT (e.g. the 0x25000000 doorbell).
    ///
    /// ponytail: identity, 4 KiB granular. If a region needs a non-identity
    /// physical target (a real MMIO doorbell whose far end must route to a device
    /// model) or a distinct attr, populate those PTEs specifically instead.
    pub fn synthesize_identity_page_table(&mut self, vaddr_lo: u32, vaddr_hi: u32, attr: u8) {
        let lo = vaddr_lo & !0xfff;
        let mut page = lo;
        while page < vaddr_hi {
            // PTE address for this 4 KiB page: PAGE_TABLE_BASE | (vaddr >> 10), word-aligned.
            let pte_addr = (PAGE_TABLE_BASE | (page >> 10)) & !0x3;
            if Self::region(pte_addr) != Region::PageTable {
                break; // past the 1 MiB PT aperture -- nothing to populate
            }
            let pte = (page & 0xffff_f000) | ((attr as u32) & 0xf);
            write_le32(&mut self.page_table, pte_addr - PAGE_TABLE_BASE, pte);
            page = page.wrapping_add(0x1000);
        }
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

    /// Pre-initialize a reconstructed D-side low-memory segment. This is a
    /// placement operation: later firmware stores remain writable.
    pub fn preload_local_data(&mut self, paddr_base: u32, data: &[u8]) {
        let off = paddr_base as usize;
        debug_assert!(off + data.len() <= LOCAL_DATA_END as usize);
        if self.local_data.len() < off + data.len() {
            self.local_data.resize(off + data.len(), 0);
        }
        self.local_data[off..off + data.len()].copy_from_slice(data);
    }

    fn local_data_offset(&self, paddr: u32) -> Option<u32> {
        (paddr < LOCAL_DATA_END && !Self::is_phoenix_device(paddr)).then_some(paddr)
    }

    /// D-side (data) load of a 32-bit word by PHYSICAL address, Harvard-routed:
    /// below [`LOCAL_DATA_END`] goes to `local_data` (DRAM); at/above it, the
    /// same region behavior as [`Bus::region_load32`] (Ram/Mailbox/PageTable
    /// backing, Array/System stubbed and probe-recorded).
    pub fn data_load32(&mut self, paddr: u32) -> u32 {
        if let Some(off) = self.local_data_offset(paddr) {
            read_le32(&self.local_data, off)
        } else {
            self.region_load32(paddr, None)
        }
    }

    /// D-side load of a single byte by physical address. See [`Bus::data_load32`].
    pub fn data_load8(&mut self, paddr: u32) -> u8 {
        if let Some(off) = self.local_data_offset(paddr) {
            byte_at(&self.local_data, off)
        } else {
            self.region_load8(paddr)
        }
    }

    /// D-side store of a 32-bit word by physical address. See [`Bus::data_load32`].
    pub fn data_store32(&mut self, paddr: u32, v: u32) {
        if let Some(off) = self.local_data_offset(paddr) {
            write_le32(&mut self.local_data, off, v);
        } else {
            self.region_store32(paddr, v, None);
        }
    }

    /// D-side store of a single byte by physical address. See [`Bus::data_load32`].
    pub fn data_store8(&mut self, paddr: u32, v: u32) {
        if let Some(off) = self.local_data_offset(paddr) {
            set_byte_at(&mut self.local_data, off, v as u8);
        } else {
            self.region_store8(paddr, v);
        }
    }

    /// I-side (instruction) load of a 32-bit word by physical address,
    /// Harvard-routed: below [`LOCAL_DATA_END`] reads the ROM image (at
    /// `paddr + load_offset`); at/above it, the same region behavior as
    /// [`Bus::region_load32`] (no Harvard split above the boundary).
    pub fn inst_load32(&mut self, paddr: u32) -> u32 {
        if paddr < LOCAL_DATA_END {
            read_le32(&self.rom, paddr.wrapping_add(self.load_offset))
        } else {
            self.region_load32(paddr, None)
        }
    }

    /// I-side 32-bit load that honors the low-window fetch overlays, keyed on
    /// the VIRTUAL address (like [`Bus::fetch8`]). `l32r` literal-pool reads
    /// route here: a +0x100-window function's literal pool is stored at
    /// `vaddr + file_offset` alongside its code, so reading at the base
    /// `load_offset` returns a word `0x100-0x5c` off (M2c dual-mapping -- the
    /// +0x100 window covers literal pools, not just instruction fetch).
    pub fn inst_load32_overlay(&mut self, vaddr: u32, paddr: u32) -> u32 {
        for &(lo, hi, off) in &self.rom_overlays {
            if (lo..hi).contains(&vaddr) {
                return read_le32(&self.rom, vaddr.wrapping_add(off));
            }
        }
        self.inst_load32(paddr)
    }

    /// I-side load of a single byte by physical address. See [`Bus::inst_load32`].
    /// This is the body [`Bus::fetch8`] calls for the non-overlay physical path.
    pub fn inst_load8(&mut self, paddr: u32) -> u8 {
        if paddr < LOCAL_DATA_END {
            byte_at(&self.rom, paddr.wrapping_add(self.load_offset))
        } else {
            self.region_load8(paddr)
        }
    }

    /// D-side bulk fill by physical address, Harvard-routed and byte-identical
    /// to `byte_len` successive [`Bus::data_store8`]s at consecutive paddrs.
    /// Splits `[paddr, paddr+byte_len)` at [`LOCAL_DATA_END`] and at every
    /// aperture transition point within the high span, filling each
    /// single-region sub-chunk via the matching bulk-fill helper (`fill_local`
    /// below the boundary -- preserving its zero-pattern no-grow optimization
    /// -- `fill_pattern` per region above it). `Region::System` covers three
    /// disjoint sub-ranges (the array/ram gap, the mailbox/page-table gap, and
    /// the unbounded tail past the page table) with different upper bounds, so
    /// boundaries are found by nearest known transition point rather than by
    /// matching on `Region` (which would merge those into one wrong chunk).
    /// Byte-identity to a `data_store8` loop requires `paddr` to be
    /// `pattern.len()`-aligned, since each boundary-split sub-fill restarts
    /// the pattern at phase 0.
    pub fn data_fill(&mut self, paddr: u32, pattern: &[u8], byte_len: usize) {
        self.data_fill_with_device(paddr, pattern, byte_len, None);
    }

    fn data_fill_with_device(
        &mut self,
        paddr: u32,
        pattern: &[u8],
        byte_len: usize,
        mut device: Option<&mut DeviceState>,
    ) {
        debug_assert!(matches!(pattern.len(), 1 | 2 | 4));
        debug_assert_eq!(byte_len % pattern.len(), 0);
        debug_assert_eq!(
            paddr as usize % pattern.len(),
            0,
            "data_fill start must be pattern-aligned: the boundary-split sub-fills reset to \
             pattern phase 0, so byte-identity to a data_store8 loop holds only when paddr is \
             pattern.len()-aligned (all region boundaries are 4-aligned)"
        );
        // Every non-local aperture transition point, in order.
        const BOUNDARIES: [u32; 8] = [
            PHOENIX_DEVICE_BASE,
            PHOENIX_DEVICE_END,
            ARRAY_END,
            RAM_BASE,
            MAILBOX_BASE,
            MAILBOX_END,
            PAGE_TABLE_BASE,
            PAGE_TABLE_END,
        ];
        let mut cur = paddr;
        let end = paddr.wrapping_add(byte_len as u32);
        while cur != end {
            if let Some(off) = self.local_data_offset(cur) {
                let low_end = if cur < PHOENIX_DEVICE_BASE {
                    LOCAL_DATA_END.min(PHOENIX_DEVICE_BASE)
                } else {
                    LOCAL_DATA_END
                };
                let chunk_len = (end - cur).min(low_end - cur) as usize;
                self.fill_local(off, pattern, chunk_len);
                cur = cur.wrapping_add(chunk_len as u32);
                continue;
            }

            let next_boundary = BOUNDARIES.iter().copied().find(|&b| b > cur).unwrap_or(end);
            let chunk_len = (end - cur).min(next_boundary - cur) as usize;
            if pattern.len() == 4 && Self::region(cur) == Region::Array {
                if let Some(device) = device.as_deref_mut() {
                    let value = u32::from_le_bytes(pattern.try_into().unwrap());
                    for off in (0..chunk_len).step_by(4) {
                        self.region_store32(cur.wrapping_add(off as u32), value, Some(&mut *device));
                    }
                } else {
                    self.fill_pattern(cur, pattern, chunk_len);
                }
            } else {
                self.fill_pattern(cur, pattern, chunk_len);
            }
            cur = cur.wrapping_add(chunk_len as u32);
        }
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

impl<'a> CpuBus<'a> {
    pub(crate) fn standalone(bus: &'a mut Bus) -> Self {
        Self::Standalone(bus)
    }

    pub(crate) fn bus(&mut self) -> &mut Bus {
        match self {
            Self::Standalone(bus) | Self::WithDevice { bus, .. } => bus,
        }
    }

    pub(crate) fn data_load32(&mut self, paddr: u32) -> u32 {
        match self {
            Self::Standalone(bus) => bus.data_load32(paddr),
            Self::WithDevice { bus, device } => {
                if let Some(off) = bus.local_data_offset(paddr) {
                    read_le32(&bus.local_data, off)
                } else {
                    bus.region_load32(paddr, Some(device))
                }
            }
        }
    }

    pub(crate) fn data_load8(&mut self, paddr: u32) -> u8 {
        self.bus().data_load8(paddr)
    }

    pub(crate) fn data_store32(&mut self, paddr: u32, value: u32) {
        match self {
            Self::Standalone(bus) => bus.data_store32(paddr, value),
            Self::WithDevice { bus, device } => {
                if let Some(off) = bus.local_data_offset(paddr) {
                    write_le32(&mut bus.local_data, off, value);
                } else {
                    bus.region_store32(paddr, value, Some(device));
                }
            }
        }
    }

    pub(crate) fn data_store8(&mut self, paddr: u32, value: u32) {
        self.bus().data_store8(paddr, value);
    }

    pub(crate) fn data_fill(&mut self, paddr: u32, pattern: &[u8], byte_len: usize) {
        match self {
            Self::Standalone(bus) => bus.data_fill(paddr, pattern, byte_len),
            Self::WithDevice { bus, device } => {
                bus.data_fill_with_device(paddr, pattern, byte_len, Some(&mut **device));
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
    fn phoenix_device_aperture_preempts_the_low_local_view() {
        let mut bus = Bus::new(vec![]);
        for addr in [0x0300_0000, 0x0308_0000, 0x030c_0000, 0x030f_fffc] {
            assert_eq!(Bus::region(addr), Region::System, "device address {addr:#x}");
            assert_eq!(bus.local_data_offset(addr), None, "device address {addr:#x}");
        }

        bus.arm_probe();
        bus.data_store32(0x0308_0000, 0x1122_3344);
        let accesses = bus.take_probe();
        assert_eq!(accesses.len(), 1);
        assert_eq!(accesses[0].region, Region::System);
        assert_eq!(bus.local_data_len_for_test(), 0);
    }

    #[test]
    fn rom_reads_little_endian_from_image() {
        let mut bus = Bus::new(vec![0x78, 0x56, 0x34, 0x12]); // @0
        assert_eq!(bus.inst_load32(0), 0x12345678);
    }

    #[test]
    fn fetch_overlay_reads_alternate_file_offset_by_vaddr() {
        // The firmware .text is not a single uniform file offset: a block of
        // functions is stored at file = vaddr + 0x100 while the rest is at the
        // base load_offset. A ROM overlay models that piecewise placement -- keyed
        // on the fetch VADDR so the code region, which aliases the same phys, is
        // unaffected.
        let mut rom = vec![0u8; 0x400];
        rom[0x110] = 0xCD; // base target: vaddr 0x100 + base offset 0x10
        rom[0x1c0] = 0xAB; // overlay target: vaddr 0x100 + overlay offset 0xc0
        rom[0x310] = 0xEE; // outside-overlay target: vaddr 0x300 + base offset 0x10
        let mut bus = Bus::new_with_load_offset(rom, 0x10);

        // Without an overlay, a low-window fetch of vaddr 0x100 reads file 0x110.
        assert_eq!(bus.fetch8(0x100, 0x100), 0xCD);

        bus.add_rom_overlay(0x100, 0x200, 0xc0);
        // Inside the overlay, the low-window fetch reads file 0x1c0 (overlay offset).
        assert_eq!(bus.fetch8(0x100, 0x100), 0xAB);
        // A code-region alias (different vaddr, SAME phys 0x100) is NOT overlaid --
        // it keeps the base offset. This is the collision that broke a phys-keyed
        // overlay (iter16): code-region .text and the low block share physical bytes.
        assert_eq!(bus.fetch8(0x2000_0100, 0x100), 0xCD);
        // Outside the overlay vaddr range, the physical path applies.
        assert_eq!(bus.fetch8(0x300, 0x300), 0xEE);
        // Plain low-window image read (I-side) is untouched by the fetch overlay.
        assert_eq!(bus.inst_load8(0x100), 0xCD);
    }

    #[test]
    fn ram_round_trips() {
        let mut bus = Bus::new(vec![]);
        bus.data_store32(0x08b00100, 0xcafebabe);
        assert_eq!(bus.data_load32(0x08b00100), 0xcafebabe);
    }

    #[test]
    fn mailbox_round_trips_as_ram_this_phase() {
        let mut bus = Bus::new(vec![]);
        bus.data_store32(0x27010d00, 0x11223344);
        assert_eq!(bus.data_load32(0x27010d00), 0x11223344);
    }

    #[test]
    fn array_store_is_stubbed_and_load_returns_zero() {
        let mut bus = Bus::new(vec![]);
        bus.data_store32(0x04000000, 0x12345678);
        assert_eq!(bus.data_load32(0x04000000), 0);
    }

    #[test]
    fn array_decode_matches_firmware_tile_formula() {
        // tile(col,row,reg) = ARRAY_BASE + col<<25 + row<<20 + reg.
        assert_eq!(Bus::decode_array_addr(ARRAY_BASE), (0, 0, 0));
        assert_eq!(Bus::decode_array_addr(ARRAY_BASE + (1 << 25) + (2 << 20) + 0x1F050), (1, 2, 0x1F050));
    }

    #[test]
    fn array_store_programs_borrowed_device() {
        // Seam B (M1): with a device borrowed, a firmware 32-bit store into the
        // Array aperture programs the real tile -- not the discard stub. Mirror
        // the device model's own lock-write test through the firmware bus: a
        // lock-5 write to tile(col=1,row=2) must land in the array's lock state
        // (6-bit signed lock value), proving the store reaches a real subsystem.
        let mut bus = Bus::new(vec![]);
        let mut device = crate::device::DeviceState::new_npu1();
        let addr = ARRAY_BASE + (1 << 25) + (2 << 20) + 0x1F050; // tile(1,2) lock 5
        assert_eq!(Bus::region(addr), Region::Array);
        bus.with_device(&mut device).data_store32(addr, 5);
        assert_eq!(device.array.tile(1, 2).locks[5].value, 5);

        // Read seam: a value placed in a plain tile register reads back through
        // the firmware bus (spare offset 0x70000 -- outside lock/DMA/status/debug
        // ranges, so read_register returns the raw stored word).
        device.write_tile_register(1, 2, 0x70000, 0xABCD_1234);
        let raddr = ARRAY_BASE + (1 << 25) + (2 << 20) + 0x70000;
        assert_eq!(bus.with_device(&mut device).data_load32(raddr), 0xABCD_1234);
    }

    #[test]
    fn array_stub_behavior_unchanged_without_device() {
        // No device attached -> pre-M1 stub: store dropped, load returns 0.
        let mut bus = Bus::new(vec![]);
        let addr = ARRAY_BASE + (1 << 25) + (2 << 20) + 0x1F050;
        bus.data_store32(addr, 5);
        assert_eq!(bus.data_load32(addr), 0);
    }

    #[test]
    fn system_access_is_stubbed_to_zero() {
        let mut bus = Bus::new(vec![]);
        assert_eq!(bus.data_load32(0xf7000000), 0);
        bus.data_store32(0xf7000000, 0xaaaaaaaa); // logged, no effect
        assert_eq!(bus.data_load32(0xf7000000), 0);
    }

    #[test]
    fn system_access_is_routed_through_sysstub() {
        let mut bus = Bus::new(vec![]);
        bus.data_load32(0xf7000000);
        bus.data_load8(0xf7000004);
        bus.data_store32(0xf7000008, 0x1);
        bus.data_store8(0xf700000c, 0x2);
        // All four accesses land in the shared SysStub log, visible via the
        // M1.7 diagnostic accessor.
        assert_eq!(bus.sysstub().accesses().len(), 4);
    }

    #[test]
    fn byte_access_is_little_endian_and_independent_of_word_access() {
        let mut bus = Bus::new(vec![]);
        bus.data_store8(0x08b00200, 0xab);
        bus.data_store8(0x08b00201, 0xcd);
        assert_eq!(bus.data_load8(0x08b00200), 0xab);
        assert_eq!(bus.data_load8(0x08b00201), 0xcd);
        assert_eq!(bus.data_load32(0x08b00200) & 0xffff, 0xcdab);
    }

    #[test]
    fn rom_access_honors_psp_load_offset() {
        // phys = file - L. With L = 4, physical address 0 reads image byte 4.
        let mut bus = Bus::new_with_load_offset(vec![0, 0, 0, 0, 0x78, 0x56, 0x34, 0x12], 4);
        assert_eq!(bus.inst_load32(0), 0x12345678); // phys 0 -> file 4
        assert_eq!(bus.inst_load8(1), 0x56); // phys 1 -> file 5
                                             // Bus::new keeps offset 0 (regression).
        let mut z = Bus::new(vec![0x78, 0x56, 0x34, 0x12]);
        assert_eq!(z.inst_load32(0), 0x12345678);
    }

    #[test]
    fn preload_ram_initializes_data_region() {
        let mut bus = Bus::new(vec![]);
        // Pre-load 8 bytes at the RAM base + an offset.
        bus.preload_ram(0x08b0_0010, &[0x36, 0xc1, 0x00, 0x4c, 0xde, 0xad, 0xbe, 0xef]);
        assert_eq!(bus.data_load32(0x08b0_0010), 0x4c00_c136); // little-endian of 36 c1 00 4c
        assert_eq!(bus.data_load32(0x08b0_0014), 0xefbe_adde);
        // Unwritten RAM stays zero; region routing unaffected.
        assert_eq!(bus.data_load32(0x08b0_0000), 0);
        assert_eq!(Bus::region(0x08b0_0010), Region::Ram);
        // Pre-loaded RAM is still writable (it's .data/.bss, not ROM).
        bus.data_store32(0x08b0_0010, 0x1234_5678);
        assert_eq!(bus.data_load32(0x08b0_0010), 0x1234_5678);
    }

    #[test]
    fn fill_pattern_matches_repeated_stores() {
        let mut bus = Bus::new(vec![]);
        // Byte fill into RAM.
        bus.fill_pattern(0x08b0_1000, &[0xab], 10);
        for a in 0x08b0_1000..0x08b0_100a {
            assert_eq!(bus.data_load8(a), 0xab, "byte fill @ {a:#x}");
        }
        assert_eq!(bus.data_load8(0x08b0_100a), 0, "one past the fill is untouched");
        // Word fill into RAM: 0xdeadbeef repeated, little-endian.
        bus.fill_pattern(0x08b0_2000, &0xdead_beefu32.to_le_bytes(), 8);
        assert_eq!(bus.data_load32(0x08b0_2000), 0xdead_beef);
        assert_eq!(bus.data_load32(0x08b0_2004), 0xdead_beef);
        // Rom/System fills are dropped (no panic, no effect).
        bus.fill_pattern(0x0000_1000, &[0xff], 0x1000); // Rom: dropped
        assert_eq!(bus.inst_load8(0x0000_1000), bus.inst_load8(0x0000_1000)); // no crash
    }

    #[test]
    fn page_table_aperture_round_trips() {
        let mut bus = Bus::new(vec![]);
        assert_eq!(Bus::region(0x3c08_0000), Region::PageTable);
        bus.write_page_table_word(0x3c08_0000, 0x08b0_5001);
        assert_eq!(bus.data_load32(0x3c08_0000), 0x08b0_5001);
        // Below and above the aperture is still System (regression).
        assert_eq!(Bus::region(0x3c10_0000), Region::System);
    }

    #[test]
    fn is_local_data_boundary() {
        assert!(Bus::is_local_data(0x0000_1000));
        assert!(Bus::is_local_data(0x03ff_ffff));
        assert!(!Bus::is_local_data(0x0400_0000)); // array aperture starts here
        assert!(!Bus::is_local_data(0x2000_0000)); // code region
    }

    #[test]
    fn local_data_round_trips_and_blank_past_image() {
        // "Blank" here means "past the (empty) image": `Bus::new(vec![])` has
        // nothing to preload, so the overlay stays empty and unwritten reads
        // are still 0.
        let mut bus = Bus::new(vec![]);
        // Blank on first read.
        assert_eq!(bus.inst_load32(0), 0); // note: image (I-side) path, unrelated
        assert_eq!(bus.load_local32(0x1000), 0);
        assert_eq!(bus.load_local8(0x1000), 0);
        // Round-trips.
        bus.store_local32(0x1000, 0xdead_beef);
        assert_eq!(bus.load_local32(0x1000), 0xdead_beef);
        bus.store_local8(0x2000, 0xab);
        assert_eq!(bus.load_local8(0x2000), 0xab);
    }

    #[test]
    fn store_local_does_not_touch_the_image() {
        // The anti-aliasing invariant: a local-data store at offset X leaves the
        // rom image byte X (read via the paddr Rom path) untouched. Before the
        // Harvard split, a low write corrupted the shared rom backing.
        let mut bus = Bus::new(vec![0x11, 0x22, 0x33, 0x44]); // rom bytes at paddr 0..4
        assert_eq!(bus.load_local32(0x0), 0, "unwritten DRAM starts zeroed");
        bus.store_local32(0x0, 0xffff_ffff); // local offset 0
                                             // The rom image (paddr 0) is unchanged.
        assert_eq!(bus.inst_load32(0x0), 0x4433_2211);
        // The local backing has the write.
        assert_eq!(bus.load_local32(0x0), 0xffff_ffff);
    }

    #[test]
    fn local_data_starts_zeroed_and_is_separate_from_the_image() {
        let rom = vec![0xAA, 0xBB, 0x11, 0x22, 0x33, 0x44]; // bytes 4,5 = phys 0 with L=4
        let mut bus = Bus::new_with_load_offset(rom, 4);
        assert_eq!(bus.load_local8(0x0), 0);
        assert_eq!(bus.load_local8(0x1), 0);
        assert_eq!(bus.load_local8(0x1000), 0);
        bus.store_local8(0x0, 0x99);
        assert_eq!(bus.load_local8(0x0), 0x99);
        assert_eq!(bus.inst_load8(0x0), 0x33, "rom image untouched by the local write");
    }

    #[test]
    fn fill_local_nonzero_fills_and_zero_does_not_grow() {
        let mut bus = Bus::new(vec![]);
        // Non-zero fill grows and repeats the pattern (little-endian store order).
        bus.fill_local(0x1000, &0xdead_beefu32.to_le_bytes(), 8);
        assert_eq!(bus.load_local32(0x1000), 0xdead_beef);
        assert_eq!(bus.load_local32(0x1004), 0xdead_beef);
        // A zero fill into never-written space is a no-op that reads back 0
        // WITHOUT allocating (the tail past current len reads 0 by default).
        let before = bus.local_data_len_for_test();
        bus.fill_local(0x0100_0000, &[0u8], 0x1000); // 16 MiB offset, all-zero
        let after = bus.local_data_len_for_test();
        assert_eq!(after, before, "zero fill must not grow the backing");
        assert_eq!(bus.load_local8(0x0100_0000), 0);
        // A zero fill DOES clear an already-written prefix.
        bus.store_local8(0x1000, 0x77);
        bus.fill_local(0x1000, &[0u8], 4);
        assert_eq!(bus.load_local8(0x1000), 0);
    }

    #[test]
    fn fill_local_zero_clears_initialized_prefix() {
        let mut bus = Bus::new(vec![]);
        bus.preload_local_data(0, &[0x11, 0x22, 0x33, 0x44, 0x55, 0x66]);
        assert_eq!(bus.load_local8(0x0), 0x11);
        let before = bus.local_data_len_for_test();
        bus.fill_local(0x0, &[0u8], 4);
        assert_eq!(bus.local_data_len_for_test(), before, "zero fill must not grow");
        assert_eq!(bus.load_local8(0x0), 0);
        assert_eq!(bus.load_local8(0x3), 0);
        assert_eq!(bus.load_local8(0x4), 0x55, "byte past the zero-fill is preserved");
    }

    #[test]
    fn data_side_low_hits_dram_inst_side_low_hits_image() {
        // Low paddr: D-side -> local_data (DRAM), I-side -> rom (image). Distinct backings.
        let mut bus = Bus::new(vec![0x11, 0x22, 0x33, 0x44]); // image bytes at paddr 0..4
        assert_eq!(bus.inst_load32(0x0), 0x4433_2211, "I-side low reads the image");
        bus.data_store32(0x0, 0xdead_beef);
        assert_eq!(bus.data_load32(0x0), 0xdead_beef, "D-side low reads/writes DRAM");
        assert_eq!(bus.inst_load32(0x0), 0x4433_2211, "image untouched by the D-side store");
    }

    #[test]
    fn data_and_inst_agree_on_high_addresses() {
        // No Harvard split above LOCAL_DATA_END: both families -> the same region backing.
        let mut bus = Bus::new(vec![]);
        bus.data_store32(0x08b0_0100, 0xcafe_babe); // RAM aperture
        assert_eq!(bus.data_load32(0x08b0_0100), 0xcafe_babe);
        assert_eq!(bus.inst_load32(0x08b0_0100), 0xcafe_babe, "high I-side == high D-side");
    }

    #[test]
    fn data_load_high_records_stub_like_load32() {
        // Mailbox/Array/System D-side reads still record a StubAccess (probe fidelity).
        let mut bus = Bus::new(vec![]);
        bus.arm_probe();
        bus.data_load32(0x2701_0d00); // Mailbox
        bus.data_store32(0x0400_0000, 1); // Array
        assert_eq!(bus.take_probe().len(), 2, "D-side high accesses are probe-recorded");
    }

    #[test]
    fn mailbox_irq_ack_clears_pending_bits_without_changing_enable() {
        let mut bus = Bus::new(vec![]);
        let ack = 0x2720_03b8;
        write_le32(&mut bus.mailbox, ack - MAILBOX_BASE, 0xf000);

        bus.data_store32(ack, 0x5000);
        assert_eq!(bus.data_load32(ack), 0xa000, "acknowledgements are write-one-to-clear");

        let enable = 0x2720_0308;
        bus.data_store32(enable, 0x5000);
        assert_eq!(bus.data_load32(enable), 0x5000, "the enable bank remains persistent");
    }

    #[test]
    fn host_sram_aliases_follow_firmware_config() {
        let mut bus = Bus::new(vec![]);

        // I2X slot 13: local 0x14800, 64 bytes -> device 0x030bb000.
        bus.data_store32(0x2721_00f8, 0x01f9_4800);
        bus.store_local32(0x14820, 0x5550_4e5f);
        bus.store_local32(0x1483c, 0x1234_5678);
        assert_eq!(bus.host_sram_load32(0x030b_b020), 0x5550_4e5f);
        assert_eq!(bus.host_sram_load32(0x030b_b03c), 0x1234_5678);
        assert_eq!(bus.host_sram_load32(0x030b_b040), 0, "past the configured span");

        // X2I slot 14: local 0x14000, 1024 bytes -> device 0x030bc000.
        bus.data_store32(0x2721_00bc, 0x1ff9_4000);
        bus.host_sram_store32(0x030b_c000, 0xcafe_babe);
        assert_eq!(bus.load_local32(0x14000), 0xcafe_babe);

        assert_eq!(bus.host_sram_load32(0x030b_a000), 0, "unconfigured slot");
        bus.host_sram_store32(0x030b_a000, 0xdead_beef);
        assert_eq!(bus.load_local32(0), 0, "unconfigured stores are dropped");
    }

    #[test]
    fn data_fill_is_byte_identical_to_per_byte_stores_across_boundary() {
        // Adversarial finding 1: a NON-ZERO fill spanning LOCAL_DATA_END must route
        // each side exactly as data_store8 would -- DRAM below, Array (dropped) above,
        // with NOTHING mis-written into local_data above the boundary.
        let mut fill = Bus::new(vec![]);
        let mut loop_ = Bus::new(vec![]);
        let start = LOCAL_DATA_END - 0x800;
        let len = 0x1000usize; // 0x800 DRAM + 0x800 Array
        fill.data_fill(start, &[0xcd], len);
        for i in 0..len as u32 {
            loop_.data_store8(start + i, 0xcd);
        }
        for i in 0..len as u32 {
            let a = start + i;
            assert_eq!(fill.data_load8(a), loop_.data_load8(a), "byte {a:#x} matches per-store");
        }
        // The array side is dropped: reads back 0, and nothing leaked into local_data.
        assert_eq!(fill.data_load8(LOCAL_DATA_END), 0, "array-side byte dropped, not in DRAM");
        assert_eq!(fill.load_local8(LOCAL_DATA_END), 0, "no mis-route into DRAM above the boundary");
    }

    #[test]
    fn data_fill_word_pattern_multi_hop_across_page_table_region() {
        // M2a: a w=4 word pattern spanning the whole (1 MiB) PageTable aperture
        // on both sides -- System gap (dropped) -> PageTable (filled) -> System
        // gap (dropped) -- crosses TWO boundaries (PAGE_TABLE_BASE,
        // PAGE_TABLE_END) in a single data_fill call. Both boundaries are
        // 4-aligned, same as the 4-aligned start, so the per-chunk phase-0
        // resets line up with the global pattern phase (see data_fill's doc) and
        // the whole span must be byte-identical to a data_store8 loop.
        let mut fill = Bus::new(vec![]);
        let mut loop_ = Bus::new(vec![]);
        let pattern = 0xdead_beefu32.to_le_bytes();
        let start = PAGE_TABLE_BASE - 8; // 4-aligned, in the dropped mailbox/page-table gap
        let len = (PAGE_TABLE_END - PAGE_TABLE_BASE) as usize + 16; // 8 bytes past each edge
        fill.data_fill(start, &pattern, len);
        for i in 0..len as u32 {
            loop_.data_store8(start + i, pattern[(i % 4) as usize] as u32);
        }
        for i in 0..len as u32 {
            let a = start + i;
            assert_eq!(fill.data_load8(a), loop_.data_load8(a), "byte {a:#x} matches per-store");
        }
        // Dropped gap on both sides reads back 0 (System does not back stores).
        assert_eq!(fill.data_load32(start), 0, "pre-boundary gap word dropped");
        assert_eq!(fill.data_load32(PAGE_TABLE_END), 0, "post-boundary gap word dropped");
        // The PageTable-backed middle reads the pattern at both its edges.
        assert_eq!(fill.data_load32(PAGE_TABLE_BASE), 0xdead_beef, "page-table word filled at near edge");
        assert_eq!(fill.data_load32(PAGE_TABLE_END - 4), 0xdead_beef, "page-table word filled at far edge");
    }

    #[test]
    fn data_fill_zero_does_not_grow_dram_backing() {
        // A zero fill into never-written DRAM is a no-op that reads 0 without
        // growing the sparse backing.
        let mut bus = Bus::new(vec![]);
        let before = bus.local_data_len_for_test();
        bus.data_fill(0x0100_0000, &[0u8], 0x0010_0000); // 16 MiB zero fill, low window
        assert_eq!(bus.local_data_len_for_test(), before, "zero fill must not grow DRAM");
        assert_eq!(bus.data_load8(0x0100_0000), 0);
    }
}
