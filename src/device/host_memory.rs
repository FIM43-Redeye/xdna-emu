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

use std::collections::BTreeMap;

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
        }
    }
}

impl std::error::Error for HostMemoryError {}

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

    /// Statistics
    total_bytes_written: u64,
    total_bytes_read: u64,
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
            total_bytes_written: 0,
            total_bytes_read: 0,
        }
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

    /// Get or create a page for the given address.
    fn get_or_create_page(&mut self, addr: u64) -> &mut [u8; Self::PAGE_SIZE] {
        let page_addr = addr & Self::PAGE_MASK;
        self.pages
            .entry(page_addr)
            .or_insert_with(|| Box::new([0u8; Self::PAGE_SIZE]))
    }

    /// Get a page for reading, if it exists.
    fn get_page(&self, addr: u64) -> Option<&[u8; Self::PAGE_SIZE]> {
        let page_addr = addr & Self::PAGE_MASK;
        self.pages.get(&page_addr).map(|b| b.as_ref())
    }

    /// Write a single byte.
    #[inline]
    pub fn write_u8(&mut self, addr: u64, value: u8) {
        let page = self.get_or_create_page(addr);
        let offset = (addr & (Self::PAGE_SIZE as u64 - 1)) as usize;
        page[offset] = value;
        self.total_bytes_written += 1;
    }

    /// Read a single byte.
    #[inline]
    pub fn read_u8(&self, addr: u64) -> u8 {
        if let Some(page) = self.get_page(addr) {
            let offset = (addr & (Self::PAGE_SIZE as u64 - 1)) as usize;
            page[offset]
        } else {
            0 // Unallocated memory reads as zero
        }
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

    /// Write a byte slice to memory.
    pub fn write_bytes(&mut self, addr: u64, data: &[u8]) {
        let mut current_addr = addr;
        let mut remaining = data;

        while !remaining.is_empty() {
            let page = self.get_or_create_page(current_addr);
            let offset = (current_addr & (Self::PAGE_SIZE as u64 - 1)) as usize;
            let space_in_page = Self::PAGE_SIZE - offset;
            let to_write = remaining.len().min(space_in_page);

            page[offset..offset + to_write].copy_from_slice(&remaining[..to_write]);

            current_addr += to_write as u64;
            remaining = &remaining[to_write..];
        }

        self.total_bytes_written += data.len() as u64;
    }

    /// Read bytes from memory into a buffer.
    pub fn read_bytes(&self, addr: u64, buf: &mut [u8]) {
        let mut current_addr = addr;
        let mut offset_in_buf = 0;

        while offset_in_buf < buf.len() {
            let page_offset = (current_addr & (Self::PAGE_SIZE as u64 - 1)) as usize;
            let space_in_page = Self::PAGE_SIZE - page_offset;
            let remaining = buf.len() - offset_in_buf;
            let to_read = remaining.min(space_in_page);

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
        let bytes = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, byte_len)
        };
        self.write_bytes(addr, bytes);

        // Update region stats
        if let Some(region) = self.regions.iter_mut().find(|r| r.overlaps(addr, byte_len)) {
            region.bytes_written += byte_len;
        }
    }

    /// Read a slice of values (convenience method for reading results).
    pub fn read_slice<T: Copy + Default>(&mut self, addr: u64, count: usize) -> Vec<T> {
        let byte_len = count * std::mem::size_of::<T>();
        let mut bytes = vec![0u8; byte_len];
        self.read_bytes(addr, &mut bytes);

        // Update region stats
        if let Some(region) = self.regions.iter_mut().find(|r| r.overlaps(addr, byte_len)) {
            region.bytes_read += byte_len;
        }

        let mut result = vec![T::default(); count];
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                result.as_mut_ptr() as *mut u8,
                byte_len,
            );
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

    /// Get total bytes read.
    pub fn total_bytes_read(&self) -> u64 {
        self.total_bytes_read
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
        self.total_bytes_written = 0;
        self.total_bytes_read = 0;
    }

    /// Hexdump a memory region for debugging.
    pub fn hexdump(&self, addr: u64, len: usize) -> String {
        let mut result = String::new();
        let mut buf = vec![0u8; len];
        self.read_bytes(addr, &mut buf);

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
            .field("total_bytes_written", &self.total_bytes_written)
            .field("total_bytes_read", &self.total_bytes_read)
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
}
