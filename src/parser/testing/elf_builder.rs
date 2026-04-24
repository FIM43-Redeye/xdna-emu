//! Fluent builder for minimal-valid AIE ELF byte streams.
//!
//! Produces bytes accepted by [`crate::parser::AieElf::parse`] (and
//! therefore by `goblin::elf::Elf::parse`). Layout derived from the
//! existing hand-rolled fixture in `elf.rs`:
//!
//! ```text
//! offset 0..52    ELF32 header (EM_AIE = 264, ELFCLASS32, little-endian)
//! offset 52..84   one PT_LOAD program header (32 bytes)
//! offset 128..    program bytes (r-x, vaddr 0)
//! ```
//!
//! Not covered: section-header table (e_shnum = 0), symbol tables,
//! relocations, multiple segments. The parser does not require these
//! for the minimum-valid acceptance check. Tests that exercise symbol
//! or section lookups should use real ELF fixtures.

use xdna_archspec::elf::EM_AIE;

const ELF_HEADER_SIZE: usize = 52;
const PROGRAM_HEADER_SIZE: usize = 32;
const PROGRAM_OFFSET: usize = 128;
/// Default EF_AIE_AIE2 flag (matches elf.rs fixture).
const EF_AIE_AIE2: u32 = 0x02;

/// Builds a byte stream that parses as a valid AIE ELF32.
///
/// ```ignore
/// let bytes = ElfBuilder::new().build();
/// let elf = AieElf::parse(&bytes).unwrap();
/// assert_eq!(elf.architecture(), AieArchitecture::Aie2);
/// ```
pub struct ElfBuilder {
    e_machine: u16,
    e_flags: u32,
    program_bytes: Vec<u8>,
}

impl ElfBuilder {
    /// New builder with defaults: EM_AIE, AIE2 flags, 16 bytes of zero
    /// program data (one PT_LOAD segment at vaddr 0).
    pub fn new() -> Self {
        Self {
            e_machine: EM_AIE,
            e_flags: EF_AIE_AIE2,
            program_bytes: vec![0u8; 16],
        }
    }

    /// Override the e_machine field. Use for negative-path tests that
    /// expect `AieElf::parse` to reject non-AIE machines.
    pub fn with_e_machine(mut self, m: u16) -> Self {
        self.e_machine = m;
        self
    }

    /// Override the e_flags field (architecture selector, e.g.
    /// EF_AIE_AIE2 = 0x02, EF_AIE_AIE2P = 0x04).
    pub fn with_e_flags(mut self, flags: u32) -> Self {
        self.e_flags = flags;
        self
    }

    /// Replace the PT_LOAD segment's bytes. The segment is always loaded
    /// at vaddr 0 with r-x flags.
    pub fn with_program_bytes(mut self, bytes: Vec<u8>) -> Self {
        self.program_bytes = bytes;
        self
    }

    /// Finalize: produce the ELF byte stream.
    pub fn build(self) -> Vec<u8> {
        let program_len = self.program_bytes.len();
        let total_size = PROGRAM_OFFSET + program_len;
        let mut out = vec![0u8; total_size];

        // ELF magic + ident
        out[0..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
        out[4] = 1; // ELFCLASS32
        out[5] = 1; // ELFDATA2LSB (little-endian)
        out[6] = 1; // EV_CURRENT

        // e_type = ET_EXEC (2)
        out[16..18].copy_from_slice(&2u16.to_le_bytes());
        // e_machine
        out[18..20].copy_from_slice(&self.e_machine.to_le_bytes());
        // e_version = 1
        out[20..24].copy_from_slice(&1u32.to_le_bytes());
        // e_entry = 0
        out[24..28].copy_from_slice(&0u32.to_le_bytes());
        // e_phoff = 52 (directly after header)
        out[28..32].copy_from_slice(&(ELF_HEADER_SIZE as u32).to_le_bytes());
        // e_shoff = 84 (after one program header) -- unused with e_shnum = 0
        out[32..36].copy_from_slice(
            &((ELF_HEADER_SIZE + PROGRAM_HEADER_SIZE) as u32).to_le_bytes(),
        );
        // e_flags
        out[36..40].copy_from_slice(&self.e_flags.to_le_bytes());
        // e_ehsize
        out[40..42].copy_from_slice(&(ELF_HEADER_SIZE as u16).to_le_bytes());
        // e_phentsize
        out[42..44].copy_from_slice(&(PROGRAM_HEADER_SIZE as u16).to_le_bytes());
        // e_phnum = 1
        out[44..46].copy_from_slice(&1u16.to_le_bytes());
        // e_shentsize = 40
        out[46..48].copy_from_slice(&40u16.to_le_bytes());
        // e_shnum = 0 (no section table)
        out[48..50].copy_from_slice(&0u16.to_le_bytes());
        // e_shstrndx = 0
        out[50..52].copy_from_slice(&0u16.to_le_bytes());

        // Program header at offset 52
        let ph = ELF_HEADER_SIZE;
        // p_type = PT_LOAD (1)
        out[ph..ph + 4].copy_from_slice(&1u32.to_le_bytes());
        // p_offset = PROGRAM_OFFSET
        out[ph + 4..ph + 8].copy_from_slice(&(PROGRAM_OFFSET as u32).to_le_bytes());
        // p_vaddr = 0
        out[ph + 8..ph + 12].copy_from_slice(&0u32.to_le_bytes());
        // p_paddr = 0
        out[ph + 12..ph + 16].copy_from_slice(&0u32.to_le_bytes());
        // p_filesz
        out[ph + 16..ph + 20].copy_from_slice(&(program_len as u32).to_le_bytes());
        // p_memsz
        out[ph + 20..ph + 24].copy_from_slice(&(program_len as u32).to_le_bytes());
        // p_flags = PF_R | PF_X (5)
        out[ph + 24..ph + 28].copy_from_slice(&5u32.to_le_bytes());
        // p_align = 16
        out[ph + 28..ph + 32].copy_from_slice(&16u32.to_le_bytes());

        // Program bytes
        out[PROGRAM_OFFSET..PROGRAM_OFFSET + program_len]
            .copy_from_slice(&self.program_bytes);

        out
    }
}

impl Default for ElfBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::AieElf;
    use xdna_archspec::elf::AieArchitecture;

    #[test]
    fn default_builder_parses_as_aie_elf() {
        let bytes = ElfBuilder::new().build();
        let elf = AieElf::parse(&bytes).expect("default ElfBuilder should parse");
        assert_eq!(elf.architecture(), AieArchitecture::Aie2);
        assert_eq!(elf.entry_point(), 0);
        assert_eq!(elf.flags(), EF_AIE_AIE2);
    }

    #[test]
    fn builder_with_custom_program_bytes_round_trips() {
        let program: Vec<u8> = (0u8..32).collect();
        let bytes = ElfBuilder::new()
            .with_program_bytes(program.clone())
            .build();
        let elf = AieElf::parse(&bytes).unwrap();

        let segments: Vec<_> = elf.load_segments().collect();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].data, &program[..]);
        assert_eq!(segments[0].vaddr, 0);
        assert!(segments[0].executable);
    }

    #[test]
    fn wrong_machine_is_rejected() {
        // EM_X86_64 = 0x3E
        let bytes = ElfBuilder::new().with_e_machine(0x3E).build();
        let err = AieElf::parse(&bytes).expect_err("non-AIE e_machine must be rejected");
        let msg = err.to_string();
        assert!(
            msg.contains("e_machine") && msg.contains("not an AIE ELF"),
            "unexpected error: {msg}"
        );
    }
}
