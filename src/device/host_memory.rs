//! Simulated host/DDR memory accessible via shim DMAs.
//!
//! In real hardware, the shim tiles connect to the NoC which provides
//! access to DDR memory. This module simulates that DDR for testing
//! and validation.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────┐
//! │           Host System                    │
//! │  ┌─────────────────────────────────┐    │
//! │  │         DDR Memory              │    │
//! │  │  (simulated by HostMemory)      │    │
//! │  └─────────────────────────────────┘    │
//! └──────────────────┬──────────────────────┘
//!                    │ PCIe/NoC
//!                    ▼
//! ┌─────────┬─────────┬─────────┬─────────┐
//! │ Shim(0) │ Shim(1) │ Shim(2) │ Shim(3) │  Row 0
//! │  DMA    │  DMA    │  DMA    │  DMA    │
//! └─────────┴─────────┴─────────┴─────────┘
//! ```
//!
//! # Usage
//!
//! ```
//! use xdna_emu::device::HostMemory;
//!
//! let mut mem = HostMemory::new();
//!
//! // Create named regions for debugging
//! mem.allocate_region("input", 0x1000_0000, 4096);
//! mem.allocate_region("output", 0x2000_0000, 4096);
//!
//! // Load test data
//! let input_data: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
//! mem.write_slice(0x1000_0000, &input_data);
//!
//! // After DMA transfers and computation...
//! let output: Vec<u32> = mem.read_slice(0x2000_0000, 8);
//! ```

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::ptr::NonNull;

/// Direction of data flow for a memory region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataDirection {
    /// Data flows from host to NPU (input buffer)
    Input,
    /// Data flows from NPU to host (output buffer)
    Output,
    /// Data flows both directions (ping-pong buffer, intermediate)
    Bidirectional,
}

/// A named memory region for debugging and tracking.
#[derive(Debug, Clone)]
pub struct MemoryRegion {
    /// Human-readable name (e.g., "input", "output", "weights")
    pub name: String,
    /// Base address in host address space
    pub base_address: u64,
    /// Size in bytes
    pub size: usize,
    /// Direction of data flow
    pub direction: DataDirection,
    /// Number of bytes written by host
    pub bytes_written: usize,
    /// Number of bytes read by host
    pub bytes_read: usize,
    /// Number of DMA reads from this region
    pub dma_reads: u64,
    /// Number of DMA writes to this region
    pub dma_writes: u64,
}

impl MemoryRegion {
    /// Create a new memory region.
    pub fn new(name: impl Into<String>, base_address: u64, size: usize, direction: DataDirection) -> Self {
        Self {
            name: name.into(),
            base_address,
            size,
            direction,
            bytes_written: 0,
            bytes_read: 0,
            dma_reads: 0,
            dma_writes: 0,
        }
    }

    /// Check if an address falls within this region.
    #[inline]
    pub fn contains(&self, addr: u64) -> bool {
        addr >= self.base_address && addr < self.base_address + self.size as u64
    }

    /// Check if an address range overlaps this region.
    #[inline]
    pub fn overlaps(&self, addr: u64, len: usize) -> bool {
        let end = addr.saturating_add(len as u64);
        let region_end = self.base_address.saturating_add(self.size as u64);
        addr < region_end && end > self.base_address
    }
}

/// Error type for host memory operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostMemoryError {
    /// Address is out of bounds or unallocated
    AddressNotMapped(u64),
    /// Alignment requirement not met
    AlignmentError { address: u64, required: usize },
    /// Region overlap on allocation
    RegionOverlap { new_base: u64, existing_name: String },
    /// Region not found
    RegionNotFound(String),
    /// External mapping has a null pointer, zero size, or overflowing range
    InvalidExternalRegion { base_address: u64, size: usize },
    /// External mappings may not overlap
    ExternalRegionOverlap { new_base: u64, existing_base: u64 },
    /// External unmap did not exactly match a live mapping
    ExternalRegionNotFound { base_address: u64, size: usize },
}

impl std::fmt::Display for HostMemoryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AddressNotMapped(addr) => write!(f, "Address 0x{:016x} not mapped", addr),
            Self::AlignmentError { address, required } => {
                write!(f, "Address 0x{:016x} not aligned to {} bytes", address, required)
            }
            Self::RegionOverlap { new_base, existing_name } => {
                write!(f, "Region at 0x{:016x} overlaps with '{}'", new_base, existing_name)
            }
            Self::RegionNotFound(name) => write!(f, "Region '{}' not found", name),
            Self::InvalidExternalRegion { base_address, size } => {
                write!(f, "Invalid external region at 0x{base_address:016x} with size {size}")
            }
            Self::ExternalRegionOverlap { new_base, existing_base } => {
                write!(f, "External region at 0x{new_base:016x} overlaps mapping at 0x{existing_base:016x}")
            }
            Self::ExternalRegionNotFound { base_address, size } => {
                write!(f, "No external region at 0x{base_address:016x} with size {size}")
            }
        }
    }
}

impl std::error::Error for HostMemoryError {}

#[derive(Debug)]
struct ExternalRegion {
    base_address: u64,
    size: usize,
    data: NonNull<u8>,
}

impl ExternalRegion {
    fn end(&self) -> u64 {
        self.base_address + self.size as u64
    }

    fn contains(&self, address: u64) -> bool {
        (self.base_address..self.end()).contains(&address)
    }
}

/// A device access to host memory that the IOMMU model blocked: bytes
/// `[address, address + len)` lie on no page of any mapped buffer.
///
/// Consecutive blocked bytes of one direction merge into one record, so a
/// DMA task that runs off its buffer shows up as one record per run of pages
/// outside every buffer, not one per word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockedAccess {
    pub address: u64,
    pub len: u64,
    pub write: bool,
}

/// Blocked accesses kept per memory before only the totals grow; a run that
/// produces more is already reported loudly and the first records name it.
const MAX_BLOCKED_RECORDS: usize = 4096;

#[derive(Debug, Default)]
struct BlockedLog {
    records: Vec<BlockedAccess>,
    total_bytes: u64,
    dropped_records: u64,
}

impl BlockedLog {
    fn record(&mut self, address: u64, len: u64, write: bool) {
        self.total_bytes += len;
        if let Some(last) = self.records.last_mut() {
            if last.write == write && last.address + last.len == address {
                last.len += len;
                return;
            }
        }
        if self.records.len() < MAX_BLOCKED_RECORDS {
            self.records.push(BlockedAccess { address, len, write });
        } else {
            self.dropped_records += 1;
        }
    }
}

/// Simulated host/DDR memory.
///
/// Uses sparse storage (BTreeMap) to efficiently handle large address spaces
/// without allocating the entire 64-bit range. Memory is allocated in 4KB pages.
pub struct HostMemory {
    /// Sparse storage: page_address -> page_data
    /// Page size is 4KB (0x1000 bytes)
    pages: BTreeMap<u64, Box<[u8; Self::PAGE_SIZE]>>,

    /// Named regions for debugging and tracking
    regions: Vec<MemoryRegion>,

    // ponytail: a linear scan fits QEMU's handful of RAM mappings; use a
    // BTreeMap only if measured map counts make this material.
    /// Live non-owned mappings, such as QEMU guest RAM.
    external_regions: Vec<ExternalRegion>,

    /// Total bytes written (for diagnostics)
    total_bytes_written: u64,

    /// IOMMU model: when set, a device access (`read_*`/`write_*`) to a page
    /// that no named region or external mapping touches is blocked -- a
    /// blocked write changes nothing, a blocked read returns zeros -- and
    /// recorded, as the IOMMU blocks and logs a device access to a page the
    /// process has not mapped. Host-side accesses (`host_read_bytes` /
    /// `host_write_bytes`) are never blocked. Off by default: bare memories
    /// in tests address host memory without registering regions.
    iommu_guard: bool,

    /// Accesses the guard blocked; interior mutability because reads are `&self`.
    blocked: RefCell<BlockedLog>,

    /// Shim DMA accesses at AIE addresses below the DDR aperture, which have no
    /// host address and so never reach this memory (see `record_outside_aperture`).
    outside_aperture: RefCell<BlockedLog>,
}

impl HostMemory {
    /// Page size for sparse storage (4KB, matching typical OS page size)
    pub const PAGE_SIZE: usize = 4096;

    /// Page address mask (lower 12 bits are offset)
    const PAGE_MASK: u64 = !(Self::PAGE_SIZE as u64 - 1);

    /// Create a new empty host memory.
    pub fn new() -> Self {
        Self {
            pages: BTreeMap::new(),
            regions: Vec::new(),
            external_regions: Vec::new(),
            total_bytes_written: 0,
            iommu_guard: false,
            blocked: RefCell::new(BlockedLog::default()),
            outside_aperture: RefCell::new(BlockedLog::default()),
        }
    }

    /// Turn the IOMMU model on or off (see the `iommu_guard` field).
    pub fn set_iommu_guard(&mut self, on: bool) {
        self.iommu_guard = on;
    }

    /// Whether the IOMMU model blocks accesses outside every mapped page.
    pub fn iommu_guard(&self) -> bool {
        self.iommu_guard
    }

    /// Number of blocked-access records so far (merged runs, capped).
    pub fn blocked_record_count(&self) -> usize {
        self.blocked.borrow().records.len()
    }

    /// Blocked-access records from index `from` on.
    pub fn blocked_accesses_since(&self, from: usize) -> Vec<BlockedAccess> {
        let log = self.blocked.borrow();
        log.records.get(from..).map(<[BlockedAccess]>::to_vec).unwrap_or_default()
    }

    /// Total bytes blocked, and records dropped past the cap, since the last take.
    pub fn blocked_totals(&self) -> (u64, u64) {
        let log = self.blocked.borrow();
        (log.total_bytes, log.dropped_records)
    }

    /// Record a shim DMA access whose AIE address lies below the DDR aperture.
    ///
    /// The shim addresses host memory through an aperture: host address = AIE
    /// address - `DDR_AIE_ADDR_OFFSET`, the offset firmware (for the first host
    /// arguments) and XRT add when they put a buffer address into a BD. An
    /// address below the offset has no host address, so the access is no host
    /// access and no IOMMU fault. Where such an access goes is not measured; the
    /// emulator drops a write and returns zeros for a read.
    pub fn record_outside_aperture(&self, address: u64, len: u64, write: bool) {
        self.outside_aperture.borrow_mut().record(address, len, write);
    }

    /// Number of outside-aperture records so far (merged runs, capped).
    pub fn outside_aperture_record_count(&self) -> usize {
        self.outside_aperture.borrow().records.len()
    }

    /// Outside-aperture records from index `from` on.
    pub fn outside_aperture_accesses_since(&self, from: usize) -> Vec<BlockedAccess> {
        let log = self.outside_aperture.borrow();
        log.records.get(from..).map(<[BlockedAccess]>::to_vec).unwrap_or_default()
    }

    /// Total bytes of shim DMA access recorded outside the aperture.
    pub fn outside_aperture_bytes(&self) -> u64 {
        self.outside_aperture.borrow().total_bytes
    }

    /// Take and clear the blocked-access log.
    pub fn take_blocked_accesses(&mut self) -> Vec<BlockedAccess> {
        std::mem::take(&mut *self.blocked.borrow_mut()).records
    }

    /// Whether any named region or external mapping touches the 4 KiB page at `page`.
    fn page_mapped(&self, page: u64) -> bool {
        let end = page.saturating_add(Self::PAGE_SIZE as u64);
        self.regions.iter().any(|r| r.overlaps(page, Self::PAGE_SIZE))
            || self.external_regions.iter().any(|r| r.base_address < end && r.end() > page)
    }

    /// Split `[addr, addr + len)` at page boundaries into runs that are all
    /// mapped or all unmapped: (start offset, length, mapped).
    fn page_runs(&self, addr: u64, len: usize) -> Vec<(usize, usize, bool)> {
        let mut runs: Vec<(usize, usize, bool)> = Vec::new();
        let mut done = 0usize;
        while done < len {
            let cur = addr.wrapping_add(done as u64);
            let in_page = Self::PAGE_SIZE - (cur & !Self::PAGE_MASK) as usize;
            let n = in_page.min(len - done);
            let mapped = self.page_mapped(cur & Self::PAGE_MASK);
            match runs.last_mut() {
                Some(last) if last.2 == mapped => last.1 += n,
                _ => runs.push((done, n, mapped)),
            }
            done += n;
        }
        runs
    }

    /// Map live caller-owned memory at a host address.
    ///
    /// # Safety
    ///
    /// `data` must remain valid for reads and writes of `size` bytes until the
    /// exact mapping is removed or this memory is cleared. External access
    /// must not race an emulator access to the same bytes.
    pub unsafe fn map_external(
        &mut self,
        base_address: u64,
        data: *mut u8,
        size: usize,
    ) -> Result<(), HostMemoryError> {
        let Some(data) = NonNull::new(data) else {
            return Err(HostMemoryError::InvalidExternalRegion { base_address, size });
        };
        let Some(end) = u64::try_from(size)
            .ok()
            .filter(|_| size != 0 && size <= isize::MAX as usize)
            .and_then(|size| base_address.checked_add(size))
        else {
            return Err(HostMemoryError::InvalidExternalRegion { base_address, size });
        };

        if let Some(existing) = self
            .external_regions
            .iter()
            .find(|existing| base_address < existing.end() && end > existing.base_address)
        {
            return Err(HostMemoryError::ExternalRegionOverlap {
                new_base: base_address,
                existing_base: existing.base_address,
            });
        }

        self.external_regions.push(ExternalRegion { base_address, size, data });
        self.external_regions.sort_unstable_by_key(|region| region.base_address);
        Ok(())
    }

    /// Remove the external mapping matching this exact address and size.
    pub fn unmap_external(&mut self, base_address: u64, size: usize) -> Result<(), HostMemoryError> {
        let Some(index) = self
            .external_regions
            .iter()
            .position(|region| region.base_address == base_address && region.size == size)
        else {
            return Err(HostMemoryError::ExternalRegionNotFound { base_address, size });
        };
        self.external_regions.remove(index);
        Ok(())
    }

    /// Whether a non-empty range is backed by one named region or a
    /// contiguous union of external mappings.
    pub fn contains_range(&self, address: u64, len: usize) -> bool {
        let Some(end) = u64::try_from(len)
            .ok()
            .filter(|_| len != 0)
            .and_then(|len| address.checked_add(len))
        else {
            return false;
        };

        if self.regions.iter().any(|region| {
            region.base_address <= address
                && region
                    .base_address
                    .checked_add(region.size as u64)
                    .is_some_and(|region_end| end <= region_end)
        }) {
            return true;
        }

        let mut current = address;
        while current < end {
            let Some(region) = self.external_regions.iter().find(|region| region.contains(current)) else {
                return false;
            };
            current = region.end().min(end);
        }
        true
    }

    /// Allocate a named memory region.
    ///
    /// This doesn't actually allocate memory (pages are allocated on demand),
    /// but registers the region for debugging and tracking.
    pub fn allocate_region(
        &mut self,
        name: impl Into<String>,
        base_address: u64,
        size: usize,
    ) -> Result<(), HostMemoryError> {
        self.allocate_region_with_direction(name, base_address, size, DataDirection::Bidirectional)
    }

    /// Allocate a named memory region with specified direction.
    pub fn allocate_region_with_direction(
        &mut self,
        name: impl Into<String>,
        base_address: u64,
        size: usize,
        direction: DataDirection,
    ) -> Result<(), HostMemoryError> {
        let name = name.into();

        // Check for overlaps
        for existing in &self.regions {
            if existing.overlaps(base_address, size) {
                return Err(HostMemoryError::RegionOverlap {
                    new_base: base_address,
                    existing_name: existing.name.clone(),
                });
            }
        }

        self.regions.push(MemoryRegion::new(name, base_address, size, direction));
        Ok(())
    }

    /// Get a region by name.
    pub fn region(&self, name: &str) -> Option<&MemoryRegion> {
        self.regions.iter().find(|r| r.name == name)
    }

    /// Get a mutable region by name.
    pub fn region_mut(&mut self, name: &str) -> Option<&mut MemoryRegion> {
        self.regions.iter_mut().find(|r| r.name == name)
    }

    /// Find the region containing an address.
    pub fn region_at(&self, addr: u64) -> Option<&MemoryRegion> {
        self.regions.iter().find(|r| r.contains(addr))
    }

    /// Get all regions.
    pub fn regions(&self) -> &[MemoryRegion] {
        &self.regions
    }

    /// Remove a region by its base address.
    ///
    /// Returns true if a region was found and removed, false otherwise.
    /// Note: this does NOT deallocate the underlying pages. Sparse pages
    /// remain in the BTreeMap but are harmless (they'll be overwritten if
    /// the address range is reused).
    pub fn free_region(&mut self, base_address: u64) -> bool {
        let before = self.regions.len();
        self.regions.retain(|r| r.base_address != base_address);
        self.regions.len() < before
    }

    /// Get or create a page for the given address.
    fn get_or_create_page(&mut self, addr: u64) -> &mut [u8; Self::PAGE_SIZE] {
        let page_addr = addr & Self::PAGE_MASK;
        self.pages.entry(page_addr).or_insert_with(|| Box::new([0u8; Self::PAGE_SIZE]))
    }

    /// Get a page for reading, if it exists.
    fn get_page(&self, addr: u64) -> Option<&[u8; Self::PAGE_SIZE]> {
        let page_addr = addr & Self::PAGE_MASK;
        self.pages.get(&page_addr).map(|b| b.as_ref())
    }

    /// Write a single byte.
    #[inline]
    pub fn write_u8(&mut self, addr: u64, value: u8) {
        self.write_bytes(addr, &[value]);
    }

    /// Read a single byte.
    #[inline]
    pub fn read_u8(&self, addr: u64) -> u8 {
        let mut value = [0];
        self.read_bytes(addr, &mut value);
        value[0]
    }

    /// Write a 32-bit word (little-endian).
    #[inline]
    pub fn write_u32(&mut self, addr: u64, value: u32) {
        self.write_bytes(addr, &value.to_le_bytes());
    }

    /// Read a 32-bit word (little-endian).
    #[inline]
    pub fn read_u32(&self, addr: u64) -> u32 {
        let mut buf = [0u8; 4];
        self.read_bytes(addr, &mut buf);
        u32::from_le_bytes(buf)
    }

    /// Write a 64-bit word (little-endian).
    #[inline]
    pub fn write_u64(&mut self, addr: u64, value: u64) {
        self.write_bytes(addr, &value.to_le_bytes());
    }

    /// Read a 64-bit word (little-endian).
    #[inline]
    pub fn read_u64(&self, addr: u64) -> u64 {
        let mut buf = [0u8; 8];
        self.read_bytes(addr, &mut buf);
        u64::from_le_bytes(buf)
    }

    /// Write a byte slice to memory as a device access: under the IOMMU
    /// model, bytes on unmapped pages are dropped and recorded.
    pub fn write_bytes(&mut self, addr: u64, data: &[u8]) {
        if !self.iommu_guard {
            return self.host_write_bytes(addr, data);
        }
        for (off, n, mapped) in self.page_runs(addr, data.len()) {
            let a = addr.wrapping_add(off as u64);
            if mapped {
                self.host_write_bytes(a, &data[off..off + n]);
            } else {
                self.blocked.get_mut().record(a, n as u64, true);
            }
        }
    }

    /// Read bytes as a device access: under the IOMMU model, bytes on
    /// unmapped pages read as zero and are recorded. What a blocked read
    /// returns on silicon is not observed (the reading channel's data went
    /// to a destination that also faulted); zeros is the choice here.
    pub fn read_bytes(&self, addr: u64, buf: &mut [u8]) {
        if !self.iommu_guard {
            return self.host_read_bytes(addr, buf);
        }
        for (off, n, mapped) in self.page_runs(addr, buf.len()) {
            let a = addr.wrapping_add(off as u64);
            if mapped {
                self.host_read_bytes(a, &mut buf[off..off + n]);
            } else {
                buf[off..off + n].fill(0);
                self.blocked.borrow_mut().record(a, n as u64, false);
            }
        }
    }

    /// Write a byte slice to memory from the host side: never blocked.
    pub fn host_write_bytes(&mut self, addr: u64, data: &[u8]) {
        let mut current_addr = addr;
        let mut remaining = data;

        while !remaining.is_empty() {
            if let Some(region) =
                self.external_regions.iter_mut().find(|region| region.contains(current_addr))
            {
                let offset = (current_addr - region.base_address) as usize;
                let to_write = remaining.len().min(region.size - offset);
                unsafe {
                    std::ptr::copy(remaining.as_ptr(), region.data.as_ptr().add(offset), to_write);
                }
                current_addr += to_write as u64;
                remaining = &remaining[to_write..];
                continue;
            }

            let offset = (current_addr & (Self::PAGE_SIZE as u64 - 1)) as usize;
            let space_in_page = Self::PAGE_SIZE - offset;
            let next_external = self
                .external_regions
                .iter()
                .find(|region| region.base_address > current_addr)
                .map_or(remaining.len(), |region| {
                    usize::try_from(region.base_address - current_addr).unwrap_or(remaining.len())
                });
            let to_write = remaining.len().min(space_in_page).min(next_external);
            let page = self.get_or_create_page(current_addr);

            page[offset..offset + to_write].copy_from_slice(&remaining[..to_write]);

            current_addr += to_write as u64;
            remaining = &remaining[to_write..];
        }

        self.total_bytes_written += data.len() as u64;
    }

    /// Read bytes from the host side: never blocked.
    pub fn host_read_bytes(&self, addr: u64, buf: &mut [u8]) {
        let mut current_addr = addr;
        let mut offset_in_buf = 0;

        while offset_in_buf < buf.len() {
            if let Some(region) = self.external_regions.iter().find(|region| region.contains(current_addr)) {
                let offset = (current_addr - region.base_address) as usize;
                let to_read = (buf.len() - offset_in_buf).min(region.size - offset);
                unsafe {
                    std::ptr::copy(
                        region.data.as_ptr().add(offset),
                        buf[offset_in_buf..].as_mut_ptr(),
                        to_read,
                    );
                }
                current_addr += to_read as u64;
                offset_in_buf += to_read;
                continue;
            }

            let page_offset = (current_addr & (Self::PAGE_SIZE as u64 - 1)) as usize;
            let space_in_page = Self::PAGE_SIZE - page_offset;
            let remaining = buf.len() - offset_in_buf;
            let next_external = self
                .external_regions
                .iter()
                .find(|region| region.base_address > current_addr)
                .map_or(remaining, |region| {
                    usize::try_from(region.base_address - current_addr).unwrap_or(remaining)
                });
            let to_read = remaining.min(space_in_page).min(next_external);

            if let Some(page) = self.get_page(current_addr) {
                buf[offset_in_buf..offset_in_buf + to_read]
                    .copy_from_slice(&page[page_offset..page_offset + to_read]);
            } else {
                // Unallocated pages read as zero
                buf[offset_in_buf..offset_in_buf + to_read].fill(0);
            }

            current_addr += to_read as u64;
            offset_in_buf += to_read;
        }
    }

    /// Write a slice of u32 values (convenience method for test data).
    pub fn write_slice<T: Copy>(&mut self, addr: u64, data: &[T]) {
        let byte_len = std::mem::size_of_val(data);
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len) };
        self.host_write_bytes(addr, bytes);

        // Update region stats
        if let Some(region) = self.regions.iter_mut().find(|r| r.overlaps(addr, byte_len)) {
            region.bytes_written += byte_len;
        }
    }

    /// Read a slice of values (convenience method for reading results).
    pub fn read_slice<T: Copy + Default>(&mut self, addr: u64, count: usize) -> Vec<T> {
        let byte_len = count * std::mem::size_of::<T>();
        let mut bytes = vec![0u8; byte_len];
        self.host_read_bytes(addr, &mut bytes);

        // Update region stats
        if let Some(region) = self.regions.iter_mut().find(|r| r.overlaps(addr, byte_len)) {
            region.bytes_read += byte_len;
        }

        let mut result = vec![T::default(); count];
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), result.as_mut_ptr() as *mut u8, byte_len);
        }
        result
    }

    /// Record a DMA read from this memory (called by DMA engine).
    pub fn record_dma_read(&mut self, addr: u64, len: usize) {
        if let Some(region) = self.regions.iter_mut().find(|r| r.overlaps(addr, len)) {
            region.dma_reads += 1;
        }
    }

    /// Record a DMA write to this memory (called by DMA engine).
    pub fn record_dma_write(&mut self, addr: u64, len: usize) {
        if let Some(region) = self.regions.iter_mut().find(|r| r.overlaps(addr, len)) {
            region.dma_writes += 1;
        }
    }

    /// Get total bytes written.
    pub fn total_bytes_written(&self) -> u64 {
        self.total_bytes_written
    }

    /// Get number of allocated pages.
    pub fn allocated_pages(&self) -> usize {
        self.pages.len()
    }

    /// Get total allocated memory in bytes.
    pub fn allocated_bytes(&self) -> usize {
        self.pages.len() * Self::PAGE_SIZE
    }

    /// Clear all memory and regions.
    pub fn clear(&mut self) {
        self.pages.clear();
        self.regions.clear();
        self.external_regions.clear();
        self.total_bytes_written = 0;
        *self.blocked.get_mut() = BlockedLog::default();
        *self.outside_aperture.get_mut() = BlockedLog::default();
    }

    /// Hexdump a memory region for debugging.
    pub fn hexdump(&self, addr: u64, len: usize) -> String {
        let mut result = String::new();
        let mut buf = vec![0u8; len];
        self.host_read_bytes(addr, &mut buf);

        for (i, chunk) in buf.chunks(16).enumerate() {
            let line_addr = addr + (i * 16) as u64;
            result.push_str(&format!("{:016x}: ", line_addr));

            // Hex bytes
            for (j, byte) in chunk.iter().enumerate() {
                if j == 8 {
                    result.push(' ');
                }
                result.push_str(&format!("{:02x} ", byte));
            }

            // Padding for short lines
            for j in chunk.len()..16 {
                if j == 8 {
                    result.push(' ');
                }
                result.push_str("   ");
            }

            // ASCII representation
            result.push_str(" |");
            for byte in chunk {
                let c = if *byte >= 0x20 && *byte < 0x7f {
                    *byte as char
                } else {
                    '.'
                };
                result.push(c);
            }
            result.push_str("|\n");
        }

        result
    }
}

impl Default for HostMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for HostMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostMemory")
            .field("allocated_pages", &self.pages.len())
            .field("allocated_bytes", &self.allocated_bytes())
            .field("regions", &self.regions.len())
            .field("external_regions", &self.external_regions.len())
            .field("total_bytes_written", &self.total_bytes_written)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_read_write() {
        let mut mem = HostMemory::new();

        mem.write_u8(0x1000, 0xAB);
        assert_eq!(mem.read_u8(0x1000), 0xAB);

        mem.write_u32(0x2000, 0xDEADBEEF);
        assert_eq!(mem.read_u32(0x2000), 0xDEADBEEF);

        mem.write_u64(0x3000, 0xCAFEBABE_12345678);
        assert_eq!(mem.read_u64(0x3000), 0xCAFEBABE_12345678);
    }

    #[test]
    fn test_unallocated_reads_zero() {
        let mem = HostMemory::new();
        assert_eq!(mem.read_u8(0x9999_0000), 0);
        assert_eq!(mem.read_u32(0x9999_0000), 0);
    }

    #[test]
    fn test_cross_page_write() {
        let mut mem = HostMemory::new();

        // Write across page boundary (page size = 4096)
        let addr = 4094; // 2 bytes before page boundary
        let data = [0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE];
        mem.write_bytes(addr, &data);

        let mut buf = [0u8; 6];
        mem.read_bytes(addr, &mut buf);
        assert_eq!(buf, data);

        // Verify two pages were allocated
        assert_eq!(mem.allocated_pages(), 2);
    }

    #[test]
    fn test_region_allocation() {
        let mut mem = HostMemory::new();

        mem.allocate_region("input", 0x1000_0000, 4096).unwrap();
        mem.allocate_region("output", 0x2000_0000, 4096).unwrap();

        assert_eq!(mem.regions().len(), 2);
        assert!(mem.region("input").is_some());
        assert!(mem.region("nonexistent").is_none());
    }

    #[test]
    fn test_region_overlap_detection() {
        let mut mem = HostMemory::new();

        mem.allocate_region("first", 0x1000_0000, 4096).unwrap();

        // Overlapping region should fail
        let result = mem.allocate_region("second", 0x1000_0800, 4096);
        assert!(matches!(result, Err(HostMemoryError::RegionOverlap { .. })));

        // Non-overlapping should succeed
        mem.allocate_region("third", 0x1000_1000, 4096).unwrap();
    }

    #[test]
    fn test_write_read_slice() {
        let mut mem = HostMemory::new();

        let input: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        mem.write_slice(0x1000_0000, &input);

        let output: Vec<u32> = mem.read_slice(0x1000_0000, 8);
        assert_eq!(input, output);
    }

    #[test]
    fn test_region_contains() {
        let region = MemoryRegion::new("test", 0x1000, 256, DataDirection::Input);

        assert!(region.contains(0x1000));
        assert!(region.contains(0x10FF));
        assert!(!region.contains(0x0FFF));
        assert!(!region.contains(0x1100));
    }

    #[test]
    fn test_region_overlaps() {
        let region = MemoryRegion::new("test", 0x1000, 256, DataDirection::Input);
        // Region covers 0x1000 to 0x10FF

        // Fully contained
        assert!(region.overlaps(0x1050, 16));

        // Partial overlap at start (0x0F80 to 0x107F overlaps region at 0x1000-0x107F)
        assert!(region.overlaps(0x0F80, 256));

        // Partial overlap at end (0x1080 to 0x117F overlaps region at 0x1080-0x10FF)
        assert!(region.overlaps(0x1080, 256));

        // No overlap before (0x0800 to 0x08FF is before region)
        assert!(!region.overlaps(0x0800, 256));

        // Adjacent but not overlapping (0x0F00 to 0x0FFF ends exactly where region starts)
        assert!(!region.overlaps(0x0F00, 256));

        // No overlap after (0x1200 to 0x12FF is after region)
        assert!(!region.overlaps(0x1200, 256));
    }

    #[test]
    fn test_statistics() {
        let mut mem = HostMemory::new();
        mem.allocate_region("test", 0x1000_0000, 4096).unwrap();

        mem.write_bytes(0x1000_0000, &[1, 2, 3, 4]);
        assert_eq!(mem.total_bytes_written(), 4);

        let input: Vec<u32> = vec![1, 2, 3, 4];
        mem.write_slice(0x1000_0100, &input);

        let region = mem.region("test").unwrap();
        assert_eq!(region.bytes_written, 16); // 4 u32s = 16 bytes
    }

    #[test]
    fn test_hexdump() {
        let mut mem = HostMemory::new();
        mem.write_bytes(0x1000, b"Hello, World!\x00\x01\x02");

        let dump = mem.hexdump(0x1000, 16);
        assert!(dump.contains("0000000000001000:"));
        assert!(dump.contains("|Hello, World!"));
    }

    #[test]
    fn test_large_address_space() {
        let mut mem = HostMemory::new();

        // Write to high addresses (64-bit)
        mem.write_u64(0x8000_0000_0000_0000, 0xDEADBEEF);
        assert_eq!(mem.read_u64(0x8000_0000_0000_0000), 0xDEADBEEF);

        // Only one page should be allocated
        assert_eq!(mem.allocated_pages(), 1);
    }

    #[test]
    fn external_mapping_is_live_caller_owned_memory() {
        let mut mem = HostMemory::new();
        let mut bytes = [1, 2, 3, 4, 5, 6, 7, 8];

        unsafe {
            mem.map_external(0x6000_0000, bytes.as_mut_ptr(), bytes.len()).unwrap();
        }

        assert_eq!(mem.read_u32(0x6000_0000), 0x0403_0201);
        mem.write_u32(0x6000_0004, 0x0c0b_0a09);
        assert_eq!(bytes[4..], [9, 10, 11, 12]);

        bytes[0] = 0xaa;
        assert_eq!(bytes[0], 0xaa);
        assert_eq!(mem.read_u8(0x6000_0000), 0xaa);
        assert_eq!(mem.allocated_pages(), 0);
    }

    #[test]
    fn external_access_and_range_validation_cross_adjacent_mappings() {
        let mut mem = HostMemory::new();
        let mut first = [1, 2, 3, 4];
        let mut second = [5, 6, 7, 8];

        unsafe {
            mem.map_external(0x7000, first.as_mut_ptr(), first.len()).unwrap();
            mem.map_external(0x7004, second.as_mut_ptr(), second.len()).unwrap();
        }

        assert!(mem.contains_range(0x7002, 4));
        assert!(!mem.contains_range(0x7002, 7));

        mem.write_bytes(0x7002, &[0xaa, 0xbb, 0xcc, 0xdd]);
        assert_eq!(first, [1, 2, 0xaa, 0xbb]);
        assert_eq!(second, [0xcc, 0xdd, 7, 8]);

        let mut result = [0; 6];
        mem.read_bytes(0x7001, &mut result);
        assert_eq!(result, [2, 0xaa, 0xbb, 0xcc, 0xdd, 7]);
    }

    #[test]
    fn external_mapping_rejects_invalid_overlap_and_inexact_unmap() {
        let mut mem = HostMemory::new();
        let mut first = [0; 8];
        let mut second = [0; 4];

        assert!(unsafe { mem.map_external(0x8000, std::ptr::null_mut(), 4) }.is_err());
        assert!(unsafe { mem.map_external(0x8000, first.as_mut_ptr(), 0) }.is_err());
        assert!(unsafe { mem.map_external(u64::MAX - 1, first.as_mut_ptr(), 4) }.is_err());

        unsafe {
            mem.map_external(0x8000, first.as_mut_ptr(), first.len()).unwrap();
        }
        assert!(unsafe { mem.map_external(0x8004, second.as_mut_ptr(), second.len()) }.is_err());
        assert!(mem.unmap_external(0x8000, 4).is_err());
        assert!(mem.contains_range(0x8000, 8));

        mem.unmap_external(0x8000, 8).unwrap();
        assert!(!mem.contains_range(0x8000, 1));
        assert!(mem.unmap_external(0x8000, 8).is_err());
    }

    #[test]
    fn external_mapping_precedes_sparse_memory_and_unmap_restores_it() {
        let mut mem = HostMemory::new();
        let mut external = [0x22];

        mem.write_u8(0x9000, 0x11);
        unsafe {
            mem.map_external(0x9000, external.as_mut_ptr(), external.len()).unwrap();
        }

        assert_eq!(mem.read_u8(0x9000), 0x22);
        mem.unmap_external(0x9000, external.len()).unwrap();
        assert_eq!(mem.read_u8(0x9000), 0x11);
    }

    #[test]
    fn sparse_access_stops_at_an_external_mapping_boundary() {
        let mut mem = HostMemory::new();
        let mut external = [0x22; 2];

        unsafe {
            mem.map_external(0x9000, external.as_mut_ptr(), external.len()).unwrap();
        }
        mem.write_bytes(0x8ffe, &[0xaa, 0xbb, 0xcc, 0xdd]);

        assert_eq!(external, [0xcc, 0xdd]);
        let mut bytes = [0; 4];
        mem.read_bytes(0x8ffe, &mut bytes);
        assert_eq!(bytes, [0xaa, 0xbb, 0xcc, 0xdd]);
    }

    #[test]
    fn contains_range_accepts_named_regions_and_clear_forgets_external_mappings() {
        let mut mem = HostMemory::new();
        let mut external = [0x44; 8];

        mem.allocate_region("named", 0xa000, 8).unwrap();
        unsafe {
            mem.map_external(0xb000, external.as_mut_ptr(), external.len()).unwrap();
        }

        assert!(mem.contains_range(0xa002, 6));
        assert!(!mem.contains_range(0xa002, 7));
        assert!(mem.contains_range(0xb000, 8));
        assert!(!mem.contains_range(0xb000, 0));

        mem.clear();
        mem.write_u8(0xb000, 0x55);
        assert_eq!(external[0], 0x44);
        assert_eq!(mem.read_u8(0xb000), 0x55);
        assert!(mem.regions().is_empty());
    }

    // -- IOMMU model ---------------------------------------------------------------------------

    fn guarded() -> HostMemory {
        let mut mem = HostMemory::new();
        mem.allocate_region("a", 0x10_0000, 0x1000).unwrap();
        mem.allocate_region("b", 0x10_3000, 0x2000).unwrap();
        mem.set_iommu_guard(true);
        mem
    }

    #[test]
    fn guard_is_off_by_default_and_unmapped_accesses_then_land() {
        let mut mem = HostMemory::new();
        assert!(!mem.iommu_guard());
        mem.write_u32(0x5000, 7);
        assert_eq!(mem.read_u32(0x5000), 7);
        assert_eq!(mem.blocked_record_count(), 0);
    }

    #[test]
    fn a_device_write_off_every_mapped_page_changes_nothing_and_is_recorded() {
        let mut mem = guarded();
        mem.write_u32(0x10_1000, 0xDEAD_BEEF); // the page after "a": unmapped
        assert_eq!(mem.host_memory_word(0x10_1000), 0);
        assert_eq!(
            mem.blocked_accesses_since(0),
            vec![BlockedAccess { address: 0x10_1000, len: 4, write: true }]
        );
    }

    #[test]
    fn a_device_read_off_every_mapped_page_returns_zeros_and_is_recorded() {
        let mut mem = guarded();
        mem.host_write_bytes(0x10_2000, &[1, 2, 3, 4]); // host side: lands in a sparse page
        assert_eq!(mem.read_u32(0x10_2000), 0);
        assert_eq!(
            mem.blocked_accesses_since(0),
            vec![BlockedAccess { address: 0x10_2000, len: 4, write: false }]
        );
    }

    #[test]
    fn mapping_is_page_granular_as_the_iommu_maps_pages() {
        let mut mem = HostMemory::new();
        mem.allocate_region("odd", 0x20_0000, 100).unwrap();
        mem.set_iommu_guard(true);
        mem.write_u32(0x20_0F00, 5); // past the region's 100 bytes, inside its page
        assert_eq!(mem.read_u32(0x20_0F00), 5);
        assert_eq!(mem.blocked_record_count(), 0);
    }

    #[test]
    fn an_access_straddling_the_end_of_a_buffer_lands_up_to_the_page_and_no_further() {
        let mut mem = guarded();
        mem.write_bytes(0x10_0FFC, &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(mem.host_memory_word(0x10_0FFC), u32::from_le_bytes([1, 2, 3, 4]));
        assert_eq!(mem.host_memory_word(0x10_1000), 0);
        assert_eq!(
            mem.blocked_accesses_since(0),
            vec![BlockedAccess { address: 0x10_1000, len: 4, write: true }]
        );
    }

    #[test]
    fn consecutive_blocked_words_merge_and_directions_do_not() {
        let mut mem = guarded();
        for i in 0..4 {
            mem.write_u32(0x10_1000 + 4 * i, 1);
        }
        let _ = mem.read_u32(0x10_1010);
        mem.write_u32(0x10_1010, 1);
        assert_eq!(
            mem.blocked_accesses_since(0),
            vec![
                BlockedAccess { address: 0x10_1000, len: 16, write: true },
                BlockedAccess { address: 0x10_1010, len: 4, write: false },
                BlockedAccess { address: 0x10_1010, len: 4, write: true },
            ]
        );
        assert_eq!(mem.blocked_totals(), (24, 0));
        assert_eq!(mem.take_blocked_accesses().len(), 3);
        assert_eq!(mem.blocked_record_count(), 0);
    }

    #[test]
    fn host_side_accesses_are_never_blocked() {
        let mut mem = guarded();
        mem.host_write_bytes(0x10_1000, &[9; 4]);
        let mut b = [0u8; 4];
        mem.host_read_bytes(0x10_1000, &mut b);
        assert_eq!(b, [9; 4]);
        assert_eq!(mem.blocked_record_count(), 0);
    }

    #[test]
    fn external_mappings_count_as_mapped() {
        let mut backing = [0u8; 16];
        let mut mem = HostMemory::new();
        unsafe { mem.map_external(0x30_0000, backing.as_mut_ptr(), backing.len()) }.unwrap();
        mem.set_iommu_guard(true);
        mem.write_u32(0x30_0004, 0x0102_0304);
        assert_eq!(mem.blocked_record_count(), 0);
        mem.unmap_external(0x30_0000, 16).unwrap();
        assert_eq!(backing[4..8], [4, 3, 2, 1]);
    }

    impl HostMemory {
        fn host_memory_word(&self, addr: u64) -> u32 {
            let mut b = [0u8; 4];
            self.host_read_bytes(addr, &mut b);
            u32::from_le_bytes(b)
        }
    }
}
