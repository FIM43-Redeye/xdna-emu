//! AIE core ELF parser
//!
//! Parses ELF binaries for AMD AI Engine cores. AIE uses ELF32 format with
//! custom machine type EM_AIE (264) and architecture flags.
//!
//! # AIE Memory Layout
//!
//! AIE cores have separate address spaces:
//! - **Program Memory (PM)**: 0x00000000 - 0x0000FFFF (64KB, .text loads here)
//! - **Data Memory (DM)**: 0x00070000 - 0x0007FFFF (64KB per tile bank)
//!
//! # Example
//!
//! ```no_run
//! use xdna_emu::parser::AieElf;
//!
//! // Load and parse an AIE ELF file
//! let data = std::fs::read("core_0_2.elf")?;
//! let elf = AieElf::parse(&data)?;
//!
//! println!("Architecture: {:?}", elf.architecture());
//! println!("Entry point: 0x{:X}", elf.entry_point());
//!
//! for func in elf.functions() {
//!     println!("Function {}: 0x{:X}", func.name, func.address);
//! }
//! # Ok::<(), anyhow::Error>(())
//! ```

use anyhow::{anyhow, bail, Result};
use goblin::elf::{Elf, program_header::PT_LOAD};
use std::path::Path;

/// AIE machine type in ELF header (e_machine field)
/// See: llvm-aie llvm/include/llvm/BinaryFormat/ELF.h
pub const EM_AIE: u16 = 264; // 0x108

/// AIE architecture flags (e_flags field)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum AieArchitecture {
    /// AIE1 - Original AI Engine (Versal)
    Aie1 = 0x01,
    /// AIE2 - AI Engine ML (Phoenix/HawkPoint NPU)
    Aie2 = 0x02,
    /// AIE2P - AI Engine ML+ (Strix/Krackan NPU)
    Aie2P = 0x03,
    /// Unknown architecture
    Unknown = 0x00,
}

impl From<u32> for AieArchitecture {
    fn from(flags: u32) -> Self {
        match flags & 0x0F {
            0x01 => AieArchitecture::Aie1,
            0x02 => AieArchitecture::Aie2,
            0x03 => AieArchitecture::Aie2P,
            _ => AieArchitecture::Unknown,
        }
    }
}

/// Memory region type for AIE address interpretation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryRegion {
    /// Program memory (code) - 0x00000000 to 0x0000FFFF
    Program,
    /// Data memory - 0x00070000 to 0x0007FFFF
    Data,
    /// Stack region in data memory
    Stack,
    /// Unknown/unmapped region
    Unknown,
}

impl MemoryRegion {
    /// Classify an address into a memory region
    pub fn from_address(addr: u32) -> Self {
        match addr {
            0x00000000..=0x0000FFFF => MemoryRegion::Program,
            0x00070000..=0x0007FFFF => MemoryRegion::Data,
            _ => MemoryRegion::Unknown,
        }
    }
}

/// A symbol from the AIE ELF file
#[derive(Debug, Clone)]
pub struct AieSymbol {
    /// Symbol name
    pub name: String,
    /// Symbol address
    pub address: u32,
    /// Symbol size in bytes
    pub size: u32,
    /// Memory region this symbol belongs to
    pub region: MemoryRegion,
    /// Whether this is a function
    pub is_function: bool,
    /// Whether this is globally visible
    pub is_global: bool,
}

/// A loadable segment from the ELF
#[derive(Debug, Clone)]
pub struct LoadSegment<'a> {
    /// Virtual address where segment loads
    pub vaddr: u32,
    /// Memory size (may be larger than file data for BSS)
    pub memsz: u32,
    /// Raw segment data
    pub data: &'a [u8],
    /// Memory region this segment belongs to
    pub region: MemoryRegion,
    /// Segment is executable
    pub executable: bool,
    /// Segment is writable
    pub writable: bool,
}

/// Parsed AIE ELF file
pub struct AieElf<'a> {
    /// Raw ELF data
    data: &'a [u8],
    /// Parsed ELF structure
    elf: Elf<'a>,
}

impl<'a> std::fmt::Debug for AieElf<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AieElf")
            .field("architecture", &self.architecture())
            .field("entry_point", &format_args!("0x{:08X}", self.entry_point()))
            .field("flags", &format_args!("0x{:08X}", self.flags()))
            .field("data_len", &self.data.len())
            .finish()
    }
}

impl<'a> AieElf<'a> {
    /// Parse an AIE ELF from raw bytes
    pub fn parse(data: &'a [u8]) -> Result<Self> {
        let elf = Elf::parse(data)
            .map_err(|e| anyhow!("Failed to parse ELF: {}", e))?;

        // Validate it's an AIE ELF
        if elf.header.e_machine != EM_AIE {
            bail!(
                "Not an AIE ELF: machine type 0x{:X}, expected 0x{:X} (EM_AIE)",
                elf.header.e_machine,
                EM_AIE
            );
        }

        // Must be 32-bit
        if elf.is_64 {
            bail!("AIE ELF must be 32-bit, got 64-bit");
        }

        Ok(Self { data, elf })
    }

    /// Parse an AIE ELF from a file
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<(Vec<u8>, Self)> {
        let data = std::fs::read(path.as_ref())
            .map_err(|e| anyhow!("Failed to read ELF file: {}", e))?;

        // We need to return ownership of data along with the parsed ELF
        // This is a bit awkward but necessary for lifetime management
        let elf = Self::parse(unsafe {
            // SAFETY: We're extending the lifetime, but the caller owns both
            std::mem::transmute::<&[u8], &'a [u8]>(data.as_slice())
        })?;

        Ok((data, elf))
    }

    /// Get the AIE architecture from ELF flags
    pub fn architecture(&self) -> AieArchitecture {
        AieArchitecture::from(self.elf.header.e_flags)
    }

    /// Get the entry point address
    pub fn entry_point(&self) -> u32 {
        self.elf.header.e_entry as u32
    }

    /// Get raw ELF flags
    pub fn flags(&self) -> u32 {
        self.elf.header.e_flags
    }

    /// Iterate over loadable segments
    pub fn load_segments(&self) -> impl Iterator<Item = LoadSegment<'a>> + '_ {
        self.elf.program_headers.iter()
            .filter(|ph| ph.p_type == PT_LOAD && ph.p_filesz > 0)
            .map(|ph| {
                let offset = ph.p_offset as usize;
                let filesz = ph.p_filesz as usize;
                let data = &self.data[offset..offset + filesz];

                LoadSegment {
                    vaddr: ph.p_vaddr as u32,
                    memsz: ph.p_memsz as u32,
                    data,
                    region: MemoryRegion::from_address(ph.p_vaddr as u32),
                    executable: ph.p_flags & 0x1 != 0, // PF_X
                    writable: ph.p_flags & 0x2 != 0,   // PF_W
                }
            })
    }

    /// Get the .text section (program code)
    pub fn text_section(&self) -> Option<&'a [u8]> {
        self.elf.section_headers.iter()
            .find(|sh| {
                if let Some(name) = self.elf.shdr_strtab.get_at(sh.sh_name) {
                    name == ".text"
                } else {
                    false
                }
            })
            .map(|sh| {
                let offset = sh.sh_offset as usize;
                let size = sh.sh_size as usize;
                &self.data[offset..offset + size]
            })
    }

    /// Get the .text section address
    pub fn text_address(&self) -> Option<u32> {
        self.elf.section_headers.iter()
            .find(|sh| {
                if let Some(name) = self.elf.shdr_strtab.get_at(sh.sh_name) {
                    name == ".text"
                } else {
                    false
                }
            })
            .map(|sh| sh.sh_addr as u32)
    }

    /// Iterate over all symbols
    pub fn symbols(&self) -> impl Iterator<Item = AieSymbol> + '_ {
        self.elf.syms.iter().filter_map(|sym| {
            let name = self.elf.strtab.get_at(sym.st_name)?;

            // Skip empty names
            if name.is_empty() {
                return None;
            }

            Some(AieSymbol {
                name: name.to_string(),
                address: sym.st_value as u32,
                size: sym.st_size as u32,
                region: MemoryRegion::from_address(sym.st_value as u32),
                is_function: sym.is_function(),
                is_global: sym.st_bind() == goblin::elf::sym::STB_GLOBAL,
            })
        })
    }

    /// Find a symbol by name
    pub fn find_symbol(&self, name: &str) -> Option<AieSymbol> {
        self.symbols().find(|s| s.name == name)
    }

    /// Get all function symbols
    pub fn functions(&self) -> impl Iterator<Item = AieSymbol> + '_ {
        self.symbols().filter(|s| s.is_function)
    }

    /// Get data memory symbols (buffers, stack, etc.)
    pub fn data_symbols(&self) -> impl Iterator<Item = AieSymbol> + '_ {
        self.symbols().filter(|s| s.region == MemoryRegion::Data)
    }

    /// Get the compiler comment string if present
    pub fn compiler_info(&self) -> Option<&str> {
        self.elf.section_headers.iter()
            .find(|sh| {
                if let Some(name) = self.elf.shdr_strtab.get_at(sh.sh_name) {
                    name == ".comment"
                } else {
                    false
                }
            })
            .and_then(|sh| {
                let offset = sh.sh_offset as usize;
                let size = sh.sh_size as usize;
                let data = &self.data[offset..offset + size];
                // Comment section may have null bytes
                std::str::from_utf8(data).ok()
                    .or_else(|| {
                        // Try to find first null-terminated string
                        let end = data.iter().position(|&b| b == 0).unwrap_or(data.len());
                        std::str::from_utf8(&data[..end]).ok()
                    })
            })
    }

    /// Print a summary of the ELF contents
    pub fn print_summary(&self) {
        println!("AIE ELF Summary");
        println!("===============");
        println!("Architecture: {:?}", self.architecture());
        println!("Entry point: 0x{:08X}", self.entry_point());
        println!("Flags: 0x{:08X}", self.flags());

        if let Some(info) = self.compiler_info() {
            let first_line = info.lines().next().unwrap_or(info);
            println!("Compiler: {}", first_line.trim_start_matches('\0'));
        }

        println!();
        println!("Load Segments:");
        for (i, seg) in self.load_segments().enumerate() {
            println!("  [{}] 0x{:08X} - 0x{:08X} ({} bytes, {:?}{}{})",
                i,
                seg.vaddr,
                seg.vaddr + seg.memsz,
                seg.data.len(),
                seg.region,
                if seg.executable { ", X" } else { "" },
                if seg.writable { ", W" } else { "" },
            );
        }

        println!();
        println!("Symbols:");
        let mut func_count = 0;
        let mut data_count = 0;
        for sym in self.symbols() {
            if sym.is_function {
                func_count += 1;
            }
            if sym.region == MemoryRegion::Data {
                data_count += 1;
            }
        }
        println!("  Functions: {}", func_count);
        println!("  Data symbols: {}", data_count);

        println!();
        println!("Functions:");
        for sym in self.functions() {
            println!("  0x{:08X} {} ({}{}bytes)",
                sym.address,
                sym.name,
                if sym.size > 0 { format!("{} ", sym.size) } else { "".to_string() },
                if sym.is_global { "global, " } else { "" },
            );
        }

        println!();
        println!("Data Memory Symbols:");
        for sym in self.data_symbols() {
            println!("  0x{:08X} {}", sym.address, sym.name);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal valid AIE ELF header (52 bytes)
    fn make_minimal_aie_elf() -> Vec<u8> {
        let mut elf = vec![0u8; 256];

        // ELF magic
        elf[0..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);

        // ELF32, little-endian, version 1
        elf[4] = 1;  // ELFCLASS32
        elf[5] = 1;  // ELFDATA2LSB
        elf[6] = 1;  // EV_CURRENT

        // e_type = ET_EXEC (2)
        elf[16..18].copy_from_slice(&2u16.to_le_bytes());

        // e_machine = EM_AIE (264 = 0x108)
        elf[18..20].copy_from_slice(&264u16.to_le_bytes());

        // e_version = 1
        elf[20..24].copy_from_slice(&1u32.to_le_bytes());

        // e_entry = 0
        elf[24..28].copy_from_slice(&0u32.to_le_bytes());

        // e_phoff = 52 (right after header)
        elf[28..32].copy_from_slice(&52u32.to_le_bytes());

        // e_shoff = 84 (after program headers)
        elf[32..36].copy_from_slice(&84u32.to_le_bytes());

        // e_flags = EF_AIE_AIE2 (0x02)
        elf[36..40].copy_from_slice(&0x02u32.to_le_bytes());

        // e_ehsize = 52
        elf[40..42].copy_from_slice(&52u16.to_le_bytes());

        // e_phentsize = 32
        elf[42..44].copy_from_slice(&32u16.to_le_bytes());

        // e_phnum = 1
        elf[44..46].copy_from_slice(&1u16.to_le_bytes());

        // e_shentsize = 40
        elf[46..48].copy_from_slice(&40u16.to_le_bytes());

        // e_shnum = 0
        elf[48..50].copy_from_slice(&0u16.to_le_bytes());

        // e_shstrndx = 0
        elf[50..52].copy_from_slice(&0u16.to_le_bytes());

        // Program header at offset 52 (32 bytes)
        // p_type = PT_LOAD (1)
        elf[52..56].copy_from_slice(&1u32.to_le_bytes());
        // p_offset = 128
        elf[56..60].copy_from_slice(&128u32.to_le_bytes());
        // p_vaddr = 0
        elf[60..64].copy_from_slice(&0u32.to_le_bytes());
        // p_paddr = 0
        elf[64..68].copy_from_slice(&0u32.to_le_bytes());
        // p_filesz = 16
        elf[68..72].copy_from_slice(&16u32.to_le_bytes());
        // p_memsz = 16
        elf[72..76].copy_from_slice(&16u32.to_le_bytes());
        // p_flags = PF_R | PF_X (5)
        elf[76..80].copy_from_slice(&5u32.to_le_bytes());
        // p_align = 16
        elf[80..84].copy_from_slice(&16u32.to_le_bytes());

        // Code at offset 128
        elf[128..144].copy_from_slice(&[
            0x15, 0x01, 0x00, 0x40,  // Sample AIE instruction
            0x01, 0x00, 0x55, 0x00,
            0xe0, 0x0c, 0x07, 0x00,
            0x01, 0x00, 0x01, 0x00,
        ]);

        elf
    }

    #[test]
    fn test_em_aie_constant() {
        assert_eq!(EM_AIE, 264);
        assert_eq!(EM_AIE, 0x108);
    }

    #[test]
    fn test_aie_architecture_from_flags() {
        assert_eq!(AieArchitecture::from(0x01), AieArchitecture::Aie1);
        assert_eq!(AieArchitecture::from(0x02), AieArchitecture::Aie2);
        assert_eq!(AieArchitecture::from(0x03), AieArchitecture::Aie2P);
        assert_eq!(AieArchitecture::from(0x00), AieArchitecture::Unknown);
        assert_eq!(AieArchitecture::from(0x12), AieArchitecture::Aie2); // Mask test
    }

    #[test]
    fn test_memory_region_classification() {
        assert_eq!(MemoryRegion::from_address(0x00000000), MemoryRegion::Program);
        assert_eq!(MemoryRegion::from_address(0x00000100), MemoryRegion::Program);
        assert_eq!(MemoryRegion::from_address(0x0000FFFF), MemoryRegion::Program);

        assert_eq!(MemoryRegion::from_address(0x00070000), MemoryRegion::Data);
        assert_eq!(MemoryRegion::from_address(0x00078000), MemoryRegion::Data);
        assert_eq!(MemoryRegion::from_address(0x0007FFFF), MemoryRegion::Data);

        assert_eq!(MemoryRegion::from_address(0x00010000), MemoryRegion::Unknown);
        assert_eq!(MemoryRegion::from_address(0x00080000), MemoryRegion::Unknown);
    }

    #[test]
    fn test_parse_minimal_aie_elf() {
        let data = make_minimal_aie_elf();
        let elf = AieElf::parse(&data).unwrap();

        assert_eq!(elf.architecture(), AieArchitecture::Aie2);
        assert_eq!(elf.entry_point(), 0);
        assert_eq!(elf.flags(), 0x02);
    }

    #[test]
    fn test_reject_non_aie_elf() {
        let mut data = make_minimal_aie_elf();
        // Change machine type to x86-64 (0x3E)
        data[18..20].copy_from_slice(&0x3Eu16.to_le_bytes());

        let result = AieElf::parse(&data);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Not an AIE ELF"));
    }

    #[test]
    fn test_load_segments() {
        let data = make_minimal_aie_elf();
        let elf = AieElf::parse(&data).unwrap();

        let segments: Vec<_> = elf.load_segments().collect();
        assert_eq!(segments.len(), 1);

        let seg = &segments[0];
        assert_eq!(seg.vaddr, 0);
        assert_eq!(seg.memsz, 16);
        assert_eq!(seg.data.len(), 16);
        assert_eq!(seg.region, MemoryRegion::Program);
        assert!(seg.executable);
        assert!(!seg.writable);
    }

    // Integration test with real ELF file
    #[test]
    fn test_parse_real_aie_elf() {
        let test_elf = "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_objFifo/aie_arch.mlir.prj/main_core_0_2.elf";

        if !std::path::Path::new(test_elf).exists() {
            eprintln!("Skipping real ELF test: file not found");
            return;
        }

        let data = std::fs::read(test_elf).unwrap();
        let elf = AieElf::parse(&data).unwrap();

        // Verify it's AIE2
        assert_eq!(elf.architecture(), AieArchitecture::Aie2);

        // Entry point should be 0
        assert_eq!(elf.entry_point(), 0);

        // Should have a .text section
        let text = elf.text_section();
        assert!(text.is_some());
        assert!(text.unwrap().len() > 0);

        // Should have at least one load segment
        let segments: Vec<_> = elf.load_segments().collect();
        assert!(!segments.is_empty());

        // Should have __start symbol
        let start = elf.find_symbol("__start");
        assert!(start.is_some());
        assert!(start.unwrap().is_function);

        // Should have data memory symbols (objFifo buffers)
        let data_syms: Vec<_> = elf.data_symbols().collect();
        assert!(!data_syms.is_empty());

        // All data symbols should be in data memory region
        for sym in &data_syms {
            assert_eq!(sym.region, MemoryRegion::Data);
        }

        // Should have compiler info
        let info = elf.compiler_info();
        assert!(info.is_some());
        assert!(info.unwrap().contains("clang") || info.unwrap().contains("LLD"));
    }

    #[test]
    fn test_functions_iterator() {
        let test_elf = "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_objFifo/aie_arch.mlir.prj/main_core_0_2.elf";

        if !std::path::Path::new(test_elf).exists() {
            return;
        }

        let data = std::fs::read(test_elf).unwrap();
        let elf = AieElf::parse(&data).unwrap();

        let funcs: Vec<_> = elf.functions().collect();

        // Should have at least __start, main, _main_init
        assert!(funcs.len() >= 3);

        // All should be marked as functions
        for f in &funcs {
            assert!(f.is_function);
        }

        // __start should exist and be at address 0
        let start = funcs.iter().find(|f| f.name == "__start");
        assert!(start.is_some());
        assert_eq!(start.unwrap().address, 0);
    }
}
