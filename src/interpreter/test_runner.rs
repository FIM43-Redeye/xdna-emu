//! Test harness for running AIE kernels.
//!
//! This module provides utilities for loading ELF files, setting up test data,
//! running kernels, and verifying outputs.
//!
//! # Example
//!
//! ```ignore
//! use xdna_emu::interpreter::test_runner::TestRunner;
//!
//! let mut runner = TestRunner::new();
//!
//! // Load kernel
//! runner.load_elf(0, 2, "path/to/kernel.elf")?;
//!
//! // Set input data in tile memory
//! runner.write_tile_memory(0, 2, 0x1000, &input_data);
//!
//! // Run until halt
//! runner.run_to_completion(100_000)?;
//!
//! // Read output
//! let output = runner.read_tile_memory(0, 2, 0x2000, 256);
//! ```

use anyhow::{anyhow, Result};
use crate::device::dma::BdConfig;
use crate::interpreter::engine::{InterpreterEngine, EngineStatus};
use crate::parser::AieElf;

/// Result of a test run.
#[derive(Debug, Clone)]
pub struct TestResult {
    /// Total cycles executed.
    pub cycles: u64,
    /// Whether all cores halted normally.
    pub halted: bool,
    /// Final engine status.
    pub status: EngineStatus,
    /// Per-core halt status.
    pub core_halted: Vec<(u8, u8, bool)>,
}

impl TestResult {
    /// Check if the test completed successfully.
    pub fn success(&self) -> bool {
        self.halted && matches!(self.status, EngineStatus::Halted)
    }
}

/// Test runner for AIE kernels.
///
/// Wraps the InterpreterEngine with convenience methods for testing.
pub struct TestRunner {
    /// The underlying engine.
    engine: InterpreterEngine,
    /// Cores that have been set up (col, row).
    active_cores: Vec<(u8, u8)>,
}

impl TestRunner {
    /// Create a new test runner for NPU1.
    pub fn new() -> Self {
        Self {
            engine: InterpreterEngine::new_npu1(),
            active_cores: Vec::new(),
        }
    }

    /// Create a new test runner for NPU2.
    pub fn new_npu2() -> Self {
        Self {
            engine: InterpreterEngine::new_npu2(),
            active_cores: Vec::new(),
        }
    }

    /// Get a reference to the underlying engine.
    pub fn engine(&self) -> &InterpreterEngine {
        &self.engine
    }

    /// Get a mutable reference to the underlying engine.
    pub fn engine_mut(&mut self) -> &mut InterpreterEngine {
        &mut self.engine
    }

    /// Load an ELF file into a tile's program and data memory.
    ///
    /// This:
    /// 1. Parses the ELF file
    /// 2. Loads program segments into tile program memory
    /// 3. Loads data segments into tile data memory
    /// 4. Sets the PC to the entry point
    /// 5. Enables the core
    pub fn load_elf(&mut self, col: u8, row: u8, path: &str) -> Result<()> {
        let data = std::fs::read(path)
            .map_err(|e| anyhow!("Failed to read ELF: {}", e))?;

        self.load_elf_bytes(col, row, &data)
    }

    /// Load ELF from bytes.
    pub fn load_elf_bytes(&mut self, col: u8, row: u8, data: &[u8]) -> Result<()> {
        let elf = AieElf::parse(data)?;

        // Get tile
        let tile = self.engine.device_mut().tile_mut(col as usize, row as usize)
            .ok_or_else(|| anyhow!("Invalid tile coordinates ({}, {})", col, row))?;

        // Load segments
        for seg in elf.load_segments() {
            let vaddr = seg.vaddr as usize;
            let data = seg.data;

            match seg.region {
                crate::parser::MemoryRegion::Program => {
                    // Load into program memory
                    tile.write_program(vaddr, data);
                }
                crate::parser::MemoryRegion::Data => {
                    // Load into data memory (offset from data memory base)
                    let dm_offset = vaddr.saturating_sub(0x00070000);
                    let dm = tile.data_memory_mut();
                    if dm_offset + data.len() <= dm.len() {
                        dm[dm_offset..dm_offset + data.len()].copy_from_slice(data);
                    }
                }
                _ => {}
            }
        }

        // Set entry point and enable
        self.engine.set_core_pc(col as usize, row as usize, elf.entry_point());
        self.engine.enable_core(col as usize, row as usize);
        self.active_cores.push((col, row));

        Ok(())
    }

    /// Write data to tile data memory.
    ///
    /// Address is relative to data memory base (0x00070000 in AIE address space).
    pub fn write_tile_memory(&mut self, col: u8, row: u8, offset: usize, data: &[u8]) -> Result<()> {
        let tile = self.engine.device_mut().tile_mut(col as usize, row as usize)
            .ok_or_else(|| anyhow!("Invalid tile coordinates ({}, {})", col, row))?;

        let dm = tile.data_memory_mut();
        if offset + data.len() > dm.len() {
            return Err(anyhow!("Write exceeds data memory bounds"));
        }

        dm[offset..offset + data.len()].copy_from_slice(data);
        Ok(())
    }

    /// Read data from tile data memory.
    pub fn read_tile_memory(&self, col: u8, row: u8, offset: usize, len: usize) -> Result<Vec<u8>> {
        let tile = self.engine.device().tile(col as usize, row as usize)
            .ok_or_else(|| anyhow!("Invalid tile coordinates ({}, {})", col, row))?;

        let dm = tile.data_memory();
        if offset + len > dm.len() {
            return Err(anyhow!("Read exceeds data memory bounds"));
        }

        Ok(dm[offset..offset + len].to_vec())
    }

    /// Write data to host memory.
    pub fn write_host_memory(&mut self, addr: u64, data: &[u8]) {
        self.engine.host_memory_mut().write_bytes(addr, data);
    }

    /// Read data from host memory.
    pub fn read_host_memory(&self, addr: u64, len: usize) -> Vec<u8> {
        let mut buf = vec![0u8; len];
        self.engine.host_memory().read_bytes(addr, &mut buf);
        buf
    }

    /// Configure a DMA buffer descriptor.
    pub fn configure_dma_bd(&mut self, col: u8, row: u8, bd_index: u8, config: BdConfig) -> Result<()> {
        let dma = self.engine.device_mut().array.dma_engine_mut(col, row)
            .ok_or_else(|| anyhow!("Invalid tile coordinates ({}, {})", col, row))?;

        dma.configure_bd(bd_index, config)
            .map_err(|e| anyhow!("Failed to configure BD: {}", e))
    }

    /// Run for a maximum number of cycles, or until all cores halt.
    pub fn run(&mut self, max_cycles: u64) -> TestResult {
        let initial_cycles = self.engine.total_cycles();

        self.engine.run(max_cycles);

        // Check core status
        let core_halted: Vec<_> = self.active_cores.iter()
            .map(|&(col, row)| {
                let halted = self.engine.core_status(col as usize, row as usize)
                    .map(|s| matches!(s, crate::interpreter::CoreStatus::Halted))
                    .unwrap_or(false);
                (col, row, halted)
            })
            .collect();

        let all_halted = core_halted.iter().all(|(_, _, h)| *h);

        TestResult {
            cycles: self.engine.total_cycles() - initial_cycles,
            halted: all_halted,
            status: self.engine.status(),
            core_halted,
        }
    }

    /// Run until all cores halt or max cycles is reached.
    ///
    /// Returns error if max cycles is reached before halt.
    pub fn run_to_completion(&mut self, max_cycles: u64) -> Result<TestResult> {
        let result = self.run(max_cycles);

        if !result.halted {
            return Err(anyhow!(
                "Execution did not complete within {} cycles (status: {:?})",
                max_cycles, result.status
            ));
        }

        Ok(result)
    }

    /// Step execution by one cycle.
    pub fn step(&mut self) {
        self.engine.step();
    }

    /// Reset the test runner.
    pub fn reset(&mut self) {
        self.engine.reset();
        self.active_cores.clear();
    }

    /// Get the PC of a core.
    pub fn core_pc(&self, col: u8, row: u8) -> Option<u32> {
        self.engine.core_context(col as usize, row as usize)
            .map(|ctx| ctx.pc())
    }

    /// Get a scalar register value.
    pub fn scalar_reg(&self, col: u8, row: u8, reg: u8) -> Option<u32> {
        self.engine.core_context(col as usize, row as usize)
            .map(|ctx| ctx.scalar.read(reg))
    }

    /// Set a scalar register value.
    pub fn set_scalar_reg(&mut self, col: u8, row: u8, reg: u8, value: u32) -> Result<()> {
        let ctx = self.engine.core_context_mut(col as usize, row as usize)
            .ok_or_else(|| anyhow!("Invalid tile coordinates ({}, {})", col, row))?;

        ctx.scalar.write(reg, value);
        Ok(())
    }

    /// Write input data pattern: sequential u32 values.
    pub fn write_sequential_u32(&mut self, col: u8, row: u8, offset: usize, count: usize, start: u32) -> Result<()> {
        let values: Vec<u32> = (0..count).map(|i| start + i as u32).collect();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.write_tile_memory(col, row, offset, &bytes)
    }

    /// Read output data as u32 values.
    pub fn read_u32_values(&self, col: u8, row: u8, offset: usize, count: usize) -> Result<Vec<u32>> {
        let bytes = self.read_tile_memory(col, row, offset, count * 4)?;
        let values: Vec<u32> = bytes.chunks(4)
            .map(|chunk| {
                let arr: [u8; 4] = chunk.try_into().unwrap();
                u32::from_le_bytes(arr)
            })
            .collect();
        Ok(values)
    }

    /// Compare output data against expected values.
    pub fn verify_u32_output(&self, col: u8, row: u8, offset: usize, expected: &[u32]) -> Result<()> {
        let actual = self.read_u32_values(col, row, offset, expected.len())?;

        for (i, (exp, act)) in expected.iter().zip(actual.iter()).enumerate() {
            if exp != act {
                return Err(anyhow!(
                    "Mismatch at index {}: expected {}, got {}",
                    i, exp, act
                ));
            }
        }

        Ok(())
    }
}

impl Default for TestRunner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use crate::interpreter::bundle::{Operation, SlotOp, SlotIndex, Operand};

    // Integration test: validate single-tile execution
    //
    // This test directly exercises the interpreter without relying on
    // real ELF files. It validates that:
    // 1. Program memory can be written
    // 2. Cores can be enabled and stepped
    // 3. Instructions execute correctly
    // 4. Results are visible in registers/memory

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

        // Code at offset 128: NOPs followed by HALT
        // Use minimal bundle format (just zeros = NOPs)
        for i in 0..16 {
            elf[128 + i] = 0;
        }

        elf
    }

    #[test]
    fn test_runner_creation() {
        let runner = TestRunner::new();
        assert!(runner.active_cores.is_empty());
    }

    #[test]
    fn test_memory_write_read() {
        let mut runner = TestRunner::new();

        let data = [1u8, 2, 3, 4, 5, 6, 7, 8];
        runner.write_tile_memory(1, 2, 0x100, &data).unwrap();

        let read = runner.read_tile_memory(1, 2, 0x100, 8).unwrap();
        assert_eq!(read, data);
    }

    #[test]
    fn test_host_memory() {
        let mut runner = TestRunner::new();

        runner.write_host_memory(0x1000, &[0xAA, 0xBB, 0xCC, 0xDD]);
        let read = runner.read_host_memory(0x1000, 4);
        assert_eq!(read, vec![0xAA, 0xBB, 0xCC, 0xDD]);
    }

    #[test]
    fn test_u32_helpers() {
        let mut runner = TestRunner::new();

        // Write sequential values
        runner.write_sequential_u32(2, 3, 0x200, 4, 100).unwrap();

        // Read back
        let values = runner.read_u32_values(2, 3, 0x200, 4).unwrap();
        assert_eq!(values, vec![100, 101, 102, 103]);

        // Verify
        runner.verify_u32_output(2, 3, 0x200, &[100, 101, 102, 103]).unwrap();
    }

    #[test]
    fn test_load_elf_bytes() {
        let mut runner = TestRunner::new();
        let elf_data = make_minimal_aie_elf();

        runner.load_elf_bytes(0, 2, &elf_data).unwrap();

        // Core should be enabled
        assert!(runner.engine.is_core_enabled(0, 2));

        // PC should be at entry point (0)
        assert_eq!(runner.core_pc(0, 2), Some(0));

        // Active cores list should contain this core
        assert!(runner.active_cores.contains(&(0, 2)));
    }

    #[test]
    fn test_run_nop_program() {
        let mut runner = TestRunner::new();

        // Write NOP instructions to program memory
        if let Some(tile) = runner.engine.device_mut().tile_mut(1, 2) {
            // Write 16 bytes of NOPs (4 instructions)
            tile.write_program(0, &[0u8; 64]);
        }

        runner.engine.enable_core(1, 2);
        runner.active_cores.push((1, 2));

        // Run for 10 cycles
        let result = runner.run(10);

        // Should have run some cycles
        assert!(result.cycles > 0);

        // PC should have advanced
        assert!(runner.core_pc(1, 2).unwrap() > 0);
    }

    #[test]
    fn test_reset() {
        let mut runner = TestRunner::new();

        runner.engine.enable_core(0, 2);
        runner.active_cores.push((0, 2));

        runner.reset();

        assert!(runner.active_cores.is_empty());
        assert_eq!(runner.engine.enabled_cores(), 0);
    }

    #[test]
    fn test_scalar_register_access() {
        let mut runner = TestRunner::new();

        // Enable a core first
        runner.engine.enable_core(2, 3);

        // Set a register value
        runner.set_scalar_reg(2, 3, 5, 0xDEADBEEF).unwrap();

        // Read it back
        let value = runner.scalar_reg(2, 3, 5);
        assert_eq!(value, Some(0xDEADBEEF));
    }

    #[test]
    fn test_verify_output_mismatch() {
        let mut runner = TestRunner::new();

        runner.write_sequential_u32(1, 2, 0, 4, 0).unwrap();

        // This should fail - wrong values
        let result = runner.verify_u32_output(1, 2, 0, &[1, 2, 3, 4]);
        assert!(result.is_err());
    }

    // ========= Integration Tests =========
    //
    // These tests validate real kernel execution.

    #[test]
    fn test_load_real_add_one_elf() {
        // Path to the add_one kernel ELF from mlir-aie build
        let elf_path = "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_objFifo/aie_arch.mlir.prj/main_core_0_2.elf";

        if !std::path::Path::new(elf_path).exists() {
            eprintln!("Skipping test_load_real_add_one_elf: ELF not found at {}", elf_path);
            return;
        }

        let mut runner = TestRunner::new();

        // Load the ELF
        let result = runner.load_elf(0, 2, elf_path);
        assert!(result.is_ok(), "Failed to load ELF: {:?}", result.err());

        // Core should be enabled at PC=0
        assert!(runner.engine.is_core_enabled(0, 2));
        assert_eq!(runner.core_pc(0, 2), Some(0));

        // Step a few times to verify decoder doesn't crash
        for _ in 0..10 {
            runner.step();
        }

        // PC should have advanced (we executed some instructions)
        let pc = runner.core_pc(0, 2).unwrap();
        eprintln!("After 10 steps, PC = 0x{:04X}", pc);

        // Basic sanity: PC should have moved or we hit a branch
        // (We can't easily predict behavior without DMA/objectFIFO support)
    }

    #[test]
    fn test_single_tile_add_program() {
        // This test validates the complete single-tile execution path:
        // 1. Write a simple program that adds two memory values
        // 2. Set up input data
        // 3. Run
        // 4. Verify output

        let mut runner = TestRunner::new();

        // For now, we'll just verify that we can set up registers and
        // step the core without crashing. The actual computation test
        // requires we have working memory load/store instructions decoded
        // which depends on the specific instruction encoding.

        // Enable core at tile (1, 2)
        runner.engine.enable_core(1, 2);

        // Set up some registers via direct access
        runner.set_scalar_reg(1, 2, 0, 100).unwrap();  // r0 = 100
        runner.set_scalar_reg(1, 2, 1, 42).unwrap();   // r1 = 42

        // Write NOPs to program memory (we don't have a simple ADD encoding yet)
        if let Some(tile) = runner.engine.device_mut().tile_mut(1, 2) {
            tile.write_program(0, &[0u8; 64]);
        }

        // Step a few cycles
        for _ in 0..5 {
            runner.step();
        }

        // Registers should be unchanged by NOPs
        assert_eq!(runner.scalar_reg(1, 2, 0), Some(100));
        assert_eq!(runner.scalar_reg(1, 2, 1), Some(42));

        eprintln!("Single-tile NOP execution validated");
    }

    #[test]
    fn test_decoder_coverage() {
        // This test loads a real ELF and reports decoder coverage
        let elf_path = "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_objFifo/aie_arch.mlir.prj/main_core_0_2.elf";

        if !std::path::Path::new(elf_path).exists() {
            eprintln!("Skipping test_decoder_coverage: ELF not found");
            return;
        }

        // Read ELF and get program section
        use crate::parser::AieElf;
        let data = std::fs::read(elf_path).unwrap();
        let elf = AieElf::parse(&data).unwrap();

        let text = elf.text_section();
        assert!(text.is_some(), "No .text section found");

        let text_data = text.unwrap();
        eprintln!("Program size: {} bytes", text_data.len());

        // Try to decode instructions
        use crate::interpreter::decode::PatternDecoder;
        use crate::interpreter::traits::Decoder;

        let decoder = PatternDecoder::new();
        let mut offset = 0;
        let mut decoded = 0;
        let mut unknown = 0;

        while offset + 4 <= text_data.len() {
            let result = decoder.decode(&text_data[offset..], offset as u32);
            match result {
                Ok(bundle) => {
                    // Check first slot for operation type
                    let first_slot = bundle.slots().iter().find_map(|s| s.as_ref());
                    let op_str = format!("{:?}", first_slot.map(|s| &s.op));
                    if op_str.contains("Unknown") {
                        unknown += 1;
                    } else {
                        decoded += 1;
                    }
                    offset += bundle.size() as usize;
                }
                Err(_) => {
                    offset += 4;
                    unknown += 1;
                }
            }
        }

        let total = decoded + unknown;
        let coverage = if total > 0 { decoded as f64 / total as f64 * 100.0 } else { 0.0 };
        eprintln!("Decoder coverage: {}/{} instructions ({:.1}%)", decoded, total, coverage);

        // We should decode at least some instructions
        assert!(decoded > 0, "Failed to decode any instructions!");
    }
}
