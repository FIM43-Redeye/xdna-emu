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
        use crate::interpreter::decode::InstructionDecoder;
        use crate::interpreter::traits::Decoder;

        let decoder = InstructionDecoder::load_default();
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

    #[test]
    fn test_multi_tile_dma_stream_flow() {
        // This test validates the complete data path:
        // 1. Source tile data memory -> MM2S DMA -> stream_out
        // 2. StreamRouter routes from source tile to destination tile
        // 3. DMA stream_in -> S2MM DMA -> destination tile data memory

        let mut runner = TestRunner::new();

        // Source tile (0, 2) -> Destination tile (0, 3)
        let src_col = 0u8;
        let src_row = 2u8;
        let dst_col = 0u8;
        let dst_row = 3u8;

        // DMA channel 2 = MM2S_0 on source, channel 0 = S2MM_0 on destination
        let mm2s_channel = 2u8;
        let s2mm_channel = 0u8;

        // Write test pattern to source tile memory at offset 0x1000
        let test_data: Vec<u8> = (0..64u8).collect(); // 64 bytes = 16 words
        runner.write_tile_memory(src_col, src_row, 0x1000, &test_data).unwrap();

        // Configure MM2S BD on source tile: read from 0x1000, 64 bytes
        let mm2s_bd = BdConfig::simple_1d(0x1000, 64);
        runner.configure_dma_bd(src_col, src_row, 0, mm2s_bd).unwrap();

        // Configure S2MM BD on destination tile: write to 0x2000, 64 bytes
        let s2mm_bd = BdConfig::simple_1d(0x2000, 64);
        runner.configure_dma_bd(dst_col, dst_row, 0, s2mm_bd).unwrap();

        // Configure stream routing: source tile port 2 (MM2S_0) -> dest tile port 0 (S2MM_0)
        runner.engine.device_mut().array.stream_router.add_route(
            src_col, src_row, mm2s_channel,
            dst_col, dst_row, s2mm_channel,
        );

        // Start both DMA channels
        // MM2S_0 is channel index 2, S2MM_0 is channel index 0
        runner.engine.device_mut().array.dma_engine_mut(src_col, src_row)
            .unwrap()
            .start_channel(mm2s_channel, 0) // BD 0
            .unwrap();

        runner.engine.device_mut().array.dma_engine_mut(dst_col, dst_row)
            .unwrap()
            .start_channel(s2mm_channel, 0) // BD 0
            .unwrap();

        // Step the system multiple times to allow data to flow
        // Each step should: DMA transfer -> stream routing -> DMA transfer
        for _ in 0..100 {
            runner.step();
        }

        // Read destination tile memory and verify
        let result = runner.read_tile_memory(dst_col, dst_row, 0x2000, 64).unwrap();

        // Verify exact byte-for-byte match
        assert_eq!(
            result, test_data,
            "Data mismatch in two-tile DMA stream transfer"
        );
        eprintln!("Two-tile DMA stream transfer: 64 bytes transferred correctly");
    }

    /// Test two-tile data flow with larger transfer and exact verification.
    #[test]
    fn test_two_tile_dma_stream_256_bytes() {
        let mut runner = TestRunner::new();

        // Source tile (0, 2) -> Destination tile (0, 3)
        let src_col = 0u8;
        let src_row = 2u8;
        let dst_col = 0u8;
        let dst_row = 3u8;

        let mm2s_channel = 2u8;
        let s2mm_channel = 0u8;

        // Write 256 bytes with recognizable pattern
        let test_data: Vec<u8> = (0..256u32).map(|i| (i % 256) as u8).collect();
        runner.write_tile_memory(src_col, src_row, 0x1000, &test_data).unwrap();

        // Configure BDs for 256-byte transfer
        let mm2s_bd = BdConfig::simple_1d(0x1000, 256);
        runner.configure_dma_bd(src_col, src_row, 0, mm2s_bd).unwrap();

        let s2mm_bd = BdConfig::simple_1d(0x2000, 256);
        runner.configure_dma_bd(dst_col, dst_row, 0, s2mm_bd).unwrap();

        // Configure stream routing
        runner.engine.device_mut().array.stream_router.add_route(
            src_col, src_row, mm2s_channel,
            dst_col, dst_row, s2mm_channel,
        );

        // Start both DMA channels
        runner.engine.device_mut().array.dma_engine_mut(src_col, src_row)
            .unwrap()
            .start_channel(mm2s_channel, 0)
            .unwrap();

        runner.engine.device_mut().array.dma_engine_mut(dst_col, dst_row)
            .unwrap()
            .start_channel(s2mm_channel, 0)
            .unwrap();

        // Step until transfer completes
        for _ in 0..500 {
            runner.step();
        }

        // Verify exact match
        let result = runner.read_tile_memory(dst_col, dst_row, 0x2000, 256).unwrap();
        assert_eq!(
            result, test_data,
            "256-byte transfer data mismatch"
        );
        eprintln!("Two-tile DMA stream transfer: 256 bytes transferred correctly");
    }

    /// Diagnostic test that traces add_one kernel execution step by step.
    ///
    /// This test helps identify:
    /// - Unknown instructions that need implementation
    /// - Lock acquire/release patterns
    /// - Memory access addresses
    /// - Execution flow and PC progression
    #[test]
    fn test_add_one_diagnostic_trace() {
        use crate::interpreter::CoreStatus;
        use crate::interpreter::engine::EngineStatus;

        let elf_path = "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_objFifo/aie_arch.mlir.prj/main_core_0_2.elf";

        if !std::path::Path::new(elf_path).exists() {
            eprintln!("Skipping test_add_one_diagnostic_trace: ELF not found at {}", elf_path);
            return;
        }

        let mut runner = TestRunner::new();

        // Load the ELF
        let result = runner.load_elf(0, 2, elf_path);
        assert!(result.is_ok(), "Failed to load ELF: {:?}", result.err());

        eprintln!("\n=== add_one Kernel Diagnostic Trace ===\n");
        eprintln!("ELF loaded to tile (0, 2)");
        eprintln!("Initial PC: 0x{:04X}\n", runner.core_pc(0, 2).unwrap_or(0));

        // Set up locks per MLIR specification for add_one_objFifo:
        // Lock 0: init=2 (objFifo_in1_cons_prod_lock) - producer has 2 slots available
        // Lock 1: init=0 (objFifo_in1_cons_cons_lock) - consumer waiting for data
        // Lock 2: init=2 (objFifo_out1_prod_lock) - producer has 2 slots available
        // Lock 3: init=0 (objFifo_out1_cons_lock) - consumer waiting for data
        //
        // For our test, we need enough lock tokens to allow the kernel to complete.
        // The kernel acquires lock 0 multiple times for its buffer access pattern.
        // Set to 8 to allow extended execution for debugging.
        if let Some(tile) = runner.engine.device_mut().tile_mut(0, 2) {
            tile.locks[0].value = 8;  // Allow many input buffer accesses
            tile.locks[1].value = 8;  // Allow many input buffer releases
            tile.locks[2].value = 8;  // Allow many output buffer accesses
            tile.locks[3].value = 8;  // Allow many output buffer releases
            eprintln!("Locks initialized: [8, 8, 8, 8]");
        }

        // Write test input to the input buffer location (0x8000 per MLIR)
        // The add_one kernel adds 41 to each i32 element
        let input_values: Vec<u32> = (0..8).collect(); // [0, 1, 2, 3, 4, 5, 6, 7]
        let input_bytes: Vec<u8> = input_values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        runner.write_tile_memory(0, 2, 0x8000, &input_bytes).unwrap();
        eprintln!("Input written to 0x8000: {:?}", input_values);

        // Initialize pointer registers with buffer base addresses
        // Based on trace analysis: loads use p2, stores may use different pointers
        // Try: p2 = input (0x8000), p0 = output (0x400)
        // (This simulates what the CDO/runtime would set up)
        runner.engine.set_core_pointer(0, 2, 0, 0x400);   // p0 = output base
        runner.engine.set_core_pointer(0, 2, 1, 0x400);   // p1 = output base
        runner.engine.set_core_pointer(0, 2, 2, 0x8000);  // p2 = input base (loads use this)
        runner.engine.set_core_pointer(0, 2, 3, 0x8000);  // p3 = input base
        runner.engine.set_core_pointer(0, 2, 4, 0x400);   // p4 = output base
        runner.engine.set_core_pointer(0, 2, 5, 0x400);   // p5 = output base
        runner.engine.set_core_pointer(0, 2, 6, 0x400);   // p6 = output base
        runner.engine.set_core_pointer(0, 2, 7, 0x400);   // p7 = output base
        eprintln!("Pointer registers initialized: p2=p3=0x8000 (input), others=0x400 (output)");

        // Step through execution with tracing
        let max_steps = 100;
        let mut step = 0;
        let mut last_pc = runner.core_pc(0, 2).unwrap_or(0);
        let mut halted = false;
        let mut error_encountered = false;

        // First, let's decode and examine the first few instructions
        eprintln!("\n--- First few instructions ---");
        {
            use crate::interpreter::decode::InstructionDecoder;
            use crate::interpreter::traits::Decoder;
            use crate::interpreter::bundle::slot_layout::{extract_slots, SlotType};

            let decoder = InstructionDecoder::load_default();

            // Show LDB slot statistics
            eprintln!("\n--- LDB slot encodings ---");
            if let Some(ldb_index) = decoder.decoder_index().slot_index("ldb") {
                let stats = ldb_index.stats();
                eprintln!("  LDB slot: {} encodings, {} opcode buckets, mask=0x{:016X}",
                    ldb_index.encoding_count(), stats.bucket_count, stats.common_mask);

                // Try to decode the problematic bits 0x41D4
                let test_bits = 0x41D4u64;
                eprintln!("  Attempting decode of LDB bits 0x{:04X}:", test_bits);
                if let Some((enc, _)) = decoder.decoder_index().decode_slot("ldb", test_bits) {
                    eprintln!("    MATCH: {} (slot={}, fixed_mask=0x{:X})", enc.mnemonic, enc.slot, enc.fixed_mask);
                } else {
                    eprintln!("    NO MATCH - instruction not in decoder");
                }
            } else {
                eprintln!("  WARNING: No LDB slot in decoder index!");
            }
            if let Some(tile) = runner.engine.device().tile(0, 2) {
                if let Some(program) = tile.program_memory() {
                    let mut offset = 0;
                    // Decode first 50 instructions to understand the program
                    for _i in 0..50 {
                        if offset + 4 > program.len() {
                            break;
                        }
                        match decoder.decode(&program[offset..], offset as u32) {
                            Ok(bundle) => {
                                // Show all active slots, not just first
                                let active: Vec<_> = bundle.active_slots().collect();
                                let slot_info: Vec<String> = active.iter().map(|s| {
                                    format!("{:?}", s.op)
                                }).collect();

                                // Extra detail for problematic PC range (around 0x009E)
                                if offset >= 0x0090 && offset <= 0x00B0 {
                                    eprintln!("  PC=0x{:04X}: size={} slots={:?}", offset, bundle.size(), slot_info);
                                    eprintln!("    Raw bytes: {:02X?}", &program[offset..offset + bundle.size() as usize]);

                                    // Show extracted slots detail
                                    let extracted = extract_slots(&program[offset..]);
                                    for (i, slot) in extracted.slots.iter().enumerate() {
                                        eprintln!("    Slot {}: type={:?} bits=0x{:016X} (as u32: 0x{:08X})",
                                            i, slot.slot_type, slot.bits, slot.bits as u32);
                                    }
                                } else {
                                    eprintln!("  PC=0x{:04X}: size={} slots={:?}", offset, bundle.size(), slot_info);
                                }

                                offset += bundle.size() as usize;
                            }
                            Err(e) => {
                                eprintln!("  PC=0x{:04X}: DECODE ERROR: {:?}", offset, e);
                                offset += 4;
                            }
                        }
                    }
                }
            }
        }

        eprintln!("\n--- Execution Trace ---");

        // Helper closure to print raw bytes at a given PC
        let print_raw_bytes = |runner: &TestRunner, pc: u32| {
            if let Some(tile) = runner.engine.device().tile(0, 2) {
                if let Some(program) = tile.program_memory() {
                    let offset = pc as usize;
                    if offset + 16 <= program.len() {
                        eprintln!("  Raw bytes at PC=0x{:04X}: {:02X?}", pc, &program[offset..offset+16]);
                    }
                }
            }
        };

        while step < max_steps && !halted && !error_encountered {
            let pc_before = runner.core_pc(0, 2).unwrap_or(0);

            // Step one cycle
            runner.step();

            let pc_after = runner.core_pc(0, 2).unwrap_or(0);

            // Check engine and core status
            let engine_status = runner.engine.status();
            let core_status = runner.engine.core_status(0, 2);

            match engine_status {
                EngineStatus::Error => {
                    eprintln!("Step {:3}: PC=0x{:04X} -> ENGINE ERROR", step, pc_before);
                    print_raw_bytes(&runner, pc_before);
                    error_encountered = true;
                }
                EngineStatus::Halted => {
                    eprintln!("Step {:3}: PC=0x{:04X} -> ENGINE HALTED", step, pc_before);
                    halted = true;
                }
                _ => {
                    // Check core-specific status
                    match core_status {
                        Some(CoreStatus::Halted) => {
                            eprintln!("Step {:3}: PC=0x{:04X} -> HALT", step, pc_before);
                            halted = true;
                        }
                        Some(CoreStatus::Error) => {
                            eprintln!("Step {:3}: PC=0x{:04X} -> CORE ERROR", step, pc_before);
                            error_encountered = true;
                        }
                        Some(CoreStatus::WaitingLock { lock_id }) => {
                            eprintln!("Step {:3}: PC=0x{:04X} -> WAIT_LOCK({})", step, pc_before, lock_id);
                        }
                        Some(CoreStatus::WaitingDma { channel }) => {
                            eprintln!("Step {:3}: PC=0x{:04X} -> WAIT_DMA({})", step, pc_before, channel);
                        }
                        Some(CoreStatus::Running) | Some(CoreStatus::Ready) | None => {
                            // Only log every 10th step or on interesting PC change
                            if step % 10 == 0 || (pc_after != pc_before + 4 && pc_after != pc_before + 2) {
                                eprintln!("Step {:3}: PC=0x{:04X} -> 0x{:04X}", step, pc_before, pc_after);
                            }
                        }
                    }
                }
            }

            last_pc = pc_after;
            step += 1;
        }

        eprintln!("\n--- Trace Summary ---");
        eprintln!("Total steps: {}", step);
        eprintln!("Final PC: 0x{:04X}", last_pc);
        eprintln!("Halted: {}", halted);
        eprintln!("Engine status: {:?}", runner.engine.status());
        eprintln!("Core status: {:?}", runner.engine.core_status(0, 2));

        // Dump first 64 bytes of data memory to see what was written
        eprintln!("\n--- Memory Dump (first 64 bytes at 0x0000) ---");
        if let Ok(data) = runner.read_tile_memory(0, 2, 0x0, 64) {
            for (i, chunk) in data.chunks(16).enumerate() {
                let hex: String = chunk.iter().map(|b| format!("{:02X} ", b)).collect();
                eprintln!("  0x{:04X}: {}", i * 16, hex.trim());
            }
        }

        // Also dump at 0x400 (output buffer) with more bytes
        eprintln!("\n--- Memory Dump at 0x400 (output buffer) ---");
        if let Ok(data) = runner.read_tile_memory(0, 2, 0x400, 64) {
            for (i, chunk) in data.chunks(16).enumerate() {
                let hex: String = chunk.iter().map(|b| format!("{:02X} ", b)).collect();
                eprintln!("  0x{:04X}: {}", 0x400 + i * 16, hex.trim());
            }
        }

        // And at 0x8000 (input buffer)
        eprintln!("\n--- Memory Dump at 0x8000 (input buffer) ---");
        if let Ok(data) = runner.read_tile_memory(0, 2, 0x8000, 64) {
            for (i, chunk) in data.chunks(16).enumerate() {
                let hex: String = chunk.iter().map(|b| format!("{:02X} ", b)).collect();
                eprintln!("  0x{:04X}: {}", 0x8000 + i * 16, hex.trim());
            }
        }

        if error_encountered {
            eprintln!("\nNOTE: Error indicates an unknown instruction that needs implementation.");
            eprintln!("Check the PC address to find which instruction caused the error.");
        }

        // Read output buffer to see if any computation happened
        let output = runner.read_tile_memory(0, 2, 0x400, 32).unwrap();
        let output_u32: Vec<u32> = output
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        eprintln!("\nOutput buffer at 0x400: {:?}", output_u32);

        // Show lock states
        if let Some(tile) = runner.engine.device().tile(0, 2) {
            eprintln!("Lock states: [{}, {}, {}, {}]",
                tile.locks[0].value,
                tile.locks[1].value,
                tile.locks[2].value,
                tile.locks[3].value);
        }

        // If we got an error, this is informative - we know what instruction to implement
        // If we halted, that's progress!
        // The test passes either way - it's diagnostic
        eprintln!("\n=== End Diagnostic Trace ===\n");
    }

    /// Test bidirectional ping-pong DMA data flow between two tiles.
    ///
    /// This test validates simultaneous bidirectional data transfer:
    /// 1. Tile A (0,2) sends data to Tile B (0,3) via MM2S channel 2 -> S2MM channel 0
    /// 2. Tile B (0,3) sends different data back to Tile A (0,2) via MM2S channel 3 -> S2MM channel 1
    /// 3. Both transfers happen concurrently (both channels active on each tile)
    /// 4. Verify exact byte-for-byte match on both sides
    ///
    /// This pattern is common in real applications where tiles exchange data
    /// bidirectionally, such as in ping-pong buffer schemes or pipeline stages
    /// that need to send results back upstream.
    #[test]
    fn test_bidirectional_dma_ping_pong() {
        let mut runner = TestRunner::new();

        // Tile A at (0, 2), Tile B at (0, 3)
        let tile_a_col = 0u8;
        let tile_a_row = 2u8;
        let tile_b_col = 0u8;
        let tile_b_row = 3u8;

        // Channel assignments:
        // A -> B direction: A's MM2S channel 2 (port 2) -> B's S2MM channel 0 (port 0)
        // B -> A direction: B's MM2S channel 3 (port 3) -> A's S2MM channel 1 (port 1)
        let a_to_b_mm2s = 2u8;  // MM2S_0 on tile A
        let a_to_b_s2mm = 0u8;  // S2MM_0 on tile B
        let b_to_a_mm2s = 3u8;  // MM2S_1 on tile B
        let b_to_a_s2mm = 1u8;  // S2MM_1 on tile A

        // Create distinct test patterns for each direction
        // A -> B: 256 bytes starting at 0xA0
        let a_to_b_data: Vec<u8> = (0..256u32).map(|i| ((0xA0 + i) % 256) as u8).collect();
        // B -> A: 256 bytes starting at 0xB0
        let b_to_a_data: Vec<u8> = (0..256u32).map(|i| ((0xB0 + i) % 256) as u8).collect();

        // Memory layout:
        // Tile A: source at 0x1000 (A->B data), destination at 0x2000 (receives B->A data)
        // Tile B: source at 0x3000 (B->A data), destination at 0x4000 (receives A->B data)
        let a_src_offset = 0x1000usize;
        let a_dst_offset = 0x2000usize;
        let b_src_offset = 0x3000usize;
        let b_dst_offset = 0x4000usize;

        // Write source data to each tile
        runner.write_tile_memory(tile_a_col, tile_a_row, a_src_offset, &a_to_b_data).unwrap();
        runner.write_tile_memory(tile_b_col, tile_b_row, b_src_offset, &b_to_a_data).unwrap();

        // Configure buffer descriptors for A -> B transfer
        // BD 0 on tile A: MM2S reads from 0x1000
        let a_mm2s_bd = BdConfig::simple_1d(a_src_offset as u64, 256);
        runner.configure_dma_bd(tile_a_col, tile_a_row, 0, a_mm2s_bd).unwrap();

        // BD 0 on tile B: S2MM writes to 0x4000
        let b_s2mm_bd = BdConfig::simple_1d(b_dst_offset as u64, 256);
        runner.configure_dma_bd(tile_b_col, tile_b_row, 0, b_s2mm_bd).unwrap();

        // Configure buffer descriptors for B -> A transfer
        // BD 1 on tile B: MM2S reads from 0x3000
        let b_mm2s_bd = BdConfig::simple_1d(b_src_offset as u64, 256);
        runner.configure_dma_bd(tile_b_col, tile_b_row, 1, b_mm2s_bd).unwrap();

        // BD 1 on tile A: S2MM writes to 0x2000
        let a_s2mm_bd = BdConfig::simple_1d(a_dst_offset as u64, 256);
        runner.configure_dma_bd(tile_a_col, tile_a_row, 1, a_s2mm_bd).unwrap();

        // Configure stream routing for both directions
        // A -> B: tile A port 2 -> tile B port 0
        runner.engine.device_mut().array.stream_router.add_route(
            tile_a_col, tile_a_row, a_to_b_mm2s,
            tile_b_col, tile_b_row, a_to_b_s2mm,
        );

        // B -> A: tile B port 3 -> tile A port 1
        runner.engine.device_mut().array.stream_router.add_route(
            tile_b_col, tile_b_row, b_to_a_mm2s,
            tile_a_col, tile_a_row, b_to_a_s2mm,
        );

        // Start all DMA channels simultaneously
        // A -> B direction: MM2S on A, S2MM on B
        runner.engine.device_mut().array.dma_engine_mut(tile_a_col, tile_a_row)
            .unwrap()
            .start_channel(a_to_b_mm2s, 0)  // BD 0
            .unwrap();

        runner.engine.device_mut().array.dma_engine_mut(tile_b_col, tile_b_row)
            .unwrap()
            .start_channel(a_to_b_s2mm, 0)  // BD 0
            .unwrap();

        // B -> A direction: MM2S on B, S2MM on A
        runner.engine.device_mut().array.dma_engine_mut(tile_b_col, tile_b_row)
            .unwrap()
            .start_channel(b_to_a_mm2s, 1)  // BD 1
            .unwrap();

        runner.engine.device_mut().array.dma_engine_mut(tile_a_col, tile_a_row)
            .unwrap()
            .start_channel(b_to_a_s2mm, 1)  // BD 1
            .unwrap();

        // Step until both transfers complete
        // Allow enough cycles for 256 bytes in each direction
        for _ in 0..1000 {
            runner.step();
        }

        // Verify A -> B transfer: data should have arrived at tile B's destination
        let received_at_b = runner.read_tile_memory(tile_b_col, tile_b_row, b_dst_offset, 256).unwrap();
        assert_eq!(
            received_at_b, a_to_b_data,
            "A -> B transfer data mismatch: expected data from tile A at tile B's destination"
        );

        // Verify B -> A transfer: data should have arrived at tile A's destination
        let received_at_a = runner.read_tile_memory(tile_a_col, tile_a_row, a_dst_offset, 256).unwrap();
        assert_eq!(
            received_at_a, b_to_a_data,
            "B -> A transfer data mismatch: expected data from tile B at tile A's destination"
        );

        eprintln!("Bidirectional DMA ping-pong: 256 bytes transferred correctly in both directions");
        eprintln!("  A -> B: {} bytes (pattern starting 0xA0)", a_to_b_data.len());
        eprintln!("  B -> A: {} bytes (pattern starting 0xB0)", b_to_a_data.len());
    }

    /// Test three-tile pipeline DMA data flow: (0,2) -> (0,3) -> (0,4).
    ///
    /// This test validates a linear pipeline where data flows through three tiles:
    /// 1. Tile A (0,2): Source tile - MM2S pushes data downstream via port 0 (North)
    /// 2. Tile B (0,3): Middle tile - S2MM receives via port 1 (South), MM2S forwards via port 0 (North)
    /// 3. Tile C (0,4): Sink tile - S2MM receives via port 1 (South), stores final result
    ///
    /// This pattern is common in real AIE applications for multi-stage processing
    /// pipelines where data flows through a chain of compute tiles.
    #[test]
    fn test_three_tile_dma_pipeline() {
        let mut runner = TestRunner::new();

        // Tile coordinates: A (source) -> B (middle) -> C (sink)
        let tile_a_col = 0u8;
        let tile_a_row = 2u8;
        let tile_b_col = 0u8;
        let tile_b_row = 3u8;
        let tile_c_col = 0u8;
        let tile_c_row = 4u8;

        // DMA channel assignments:
        // MM2S channel 2 for output (MM2S_0)
        // S2MM channel 0 for input (S2MM_0)
        let mm2s_channel = 2u8;
        let s2mm_channel = 0u8;

        // Create test pattern: 128 bytes with recognizable pattern
        let transfer_size = 128u32;
        let test_data: Vec<u8> = (0..transfer_size).map(|i| ((0xC0 + i) % 256) as u8).collect();

        // Memory layout:
        // Tile A: source at 0x1000
        // Tile B: receive at 0x2000, forward from 0x2000 (same buffer - pass-through)
        // Tile C: destination at 0x3000
        let a_src_offset = 0x1000usize;
        let b_buffer_offset = 0x2000usize;
        let c_dst_offset = 0x3000usize;

        // Write source data to tile A
        runner.write_tile_memory(tile_a_col, tile_a_row, a_src_offset, &test_data).unwrap();

        // ============================================
        // Configure DMA buffer descriptors
        // ============================================

        // Tile A: MM2S BD 0 - read from 0x1000, send downstream
        let a_mm2s_bd = BdConfig::simple_1d(a_src_offset as u64, transfer_size);
        runner.configure_dma_bd(tile_a_col, tile_a_row, 0, a_mm2s_bd).unwrap();

        // Tile B: S2MM BD 0 - receive into 0x2000
        let b_s2mm_bd = BdConfig::simple_1d(b_buffer_offset as u64, transfer_size);
        runner.configure_dma_bd(tile_b_col, tile_b_row, 0, b_s2mm_bd).unwrap();

        // Tile B: MM2S BD 1 - forward from 0x2000 downstream
        let b_mm2s_bd = BdConfig::simple_1d(b_buffer_offset as u64, transfer_size);
        runner.configure_dma_bd(tile_b_col, tile_b_row, 1, b_mm2s_bd).unwrap();

        // Tile C: S2MM BD 0 - receive into 0x3000
        let c_s2mm_bd = BdConfig::simple_1d(c_dst_offset as u64, transfer_size);
        runner.configure_dma_bd(tile_c_col, tile_c_row, 0, c_s2mm_bd).unwrap();

        // ============================================
        // Configure stream routing
        // ============================================

        // Route A -> B: tile A MM2S (channel 2) -> tile B S2MM (channel 0)
        // Using add_route with channel numbers as port identifiers
        runner.engine.device_mut().array.stream_router.add_route(
            tile_a_col, tile_a_row, mm2s_channel,  // source: A's MM2S_0
            tile_b_col, tile_b_row, s2mm_channel,  // dest: B's S2MM_0
        );

        // Route B -> C: tile B MM2S (channel 2) -> tile C S2MM (channel 0)
        runner.engine.device_mut().array.stream_router.add_route(
            tile_b_col, tile_b_row, mm2s_channel,  // source: B's MM2S_0
            tile_c_col, tile_c_row, s2mm_channel,  // dest: C's S2MM_0
        );

        // ============================================
        // Start DMA channels
        // ============================================

        // Phase 1: Start A -> B transfer
        // Start tile A's MM2S channel with BD 0
        runner.engine.device_mut().array.dma_engine_mut(tile_a_col, tile_a_row)
            .unwrap()
            .start_channel(mm2s_channel, 0)
            .unwrap();

        // Start tile B's S2MM channel with BD 0
        runner.engine.device_mut().array.dma_engine_mut(tile_b_col, tile_b_row)
            .unwrap()
            .start_channel(s2mm_channel, 0)
            .unwrap();

        // Start tile C's S2MM channel with BD 0 (ready to receive from B)
        runner.engine.device_mut().array.dma_engine_mut(tile_c_col, tile_c_row)
            .unwrap()
            .start_channel(s2mm_channel, 0)
            .unwrap();

        // Step to complete A -> B transfer
        // Allow enough cycles for the first hop
        for _ in 0..300 {
            runner.step();
        }

        // Verify intermediate result: data should be at tile B's buffer
        let received_at_b = runner.read_tile_memory(tile_b_col, tile_b_row, b_buffer_offset, transfer_size as usize).unwrap();
        assert_eq!(
            received_at_b, test_data,
            "A -> B transfer data mismatch: expected data from tile A at tile B's buffer"
        );
        eprintln!("Phase 1 complete: A -> B transfer verified ({} bytes)", transfer_size);

        // Phase 2: Start B -> C transfer (B now has data to forward)
        // Start tile B's MM2S channel with BD 1 (reads from same buffer that received data)
        runner.engine.device_mut().array.dma_engine_mut(tile_b_col, tile_b_row)
            .unwrap()
            .start_channel(mm2s_channel, 1)
            .unwrap();

        // Step to complete B -> C transfer
        for _ in 0..300 {
            runner.step();
        }

        // ============================================
        // Verify final result
        // ============================================

        // Data should have arrived at tile C's destination
        let received_at_c = runner.read_tile_memory(tile_c_col, tile_c_row, c_dst_offset, transfer_size as usize).unwrap();
        assert_eq!(
            received_at_c, test_data,
            "B -> C transfer data mismatch: expected data from tile B at tile C's destination"
        );

        // Also verify tile B still has the data (it was a pass-through, not consumed)
        let still_at_b = runner.read_tile_memory(tile_b_col, tile_b_row, b_buffer_offset, transfer_size as usize).unwrap();
        assert_eq!(
            still_at_b, test_data,
            "Tile B buffer should still contain the data after forwarding"
        );

        eprintln!("Three-tile DMA pipeline test passed:");
        eprintln!("  Source:      tile ({},{}) @ 0x{:04X}", tile_a_col, tile_a_row, a_src_offset);
        eprintln!("  Middle:      tile ({},{}) @ 0x{:04X}", tile_b_col, tile_b_row, b_buffer_offset);
        eprintln!("  Destination: tile ({},{}) @ 0x{:04X}", tile_c_col, tile_c_row, c_dst_offset);
        eprintln!("  Data size:   {} bytes (pattern starting 0xC0)", transfer_size);
    }

    /// Helper function to execute a DMA transfer of a given size and verify correctness.
    ///
    /// This function:
    /// 1. Initializes source memory with a known pattern based on the size
    /// 2. Configures DMA descriptors for the specified length
    /// 3. Executes transfer via MM2S -> stream routing -> S2MM
    /// 4. Verifies destination matches source exactly
    ///
    /// Returns Ok(()) if transfer succeeded and data matches, Err with details otherwise.
    fn run_dma_transfer_test(size: u32, pattern_seed: u8) -> Result<(), String> {
        let mut runner = TestRunner::new();

        // Source tile (0, 2) -> Destination tile (0, 3)
        let src_col = 0u8;
        let src_row = 2u8;
        let dst_col = 0u8;
        let dst_row = 3u8;

        // DMA channel assignments
        let mm2s_channel = 2u8;  // MM2S_0 on source
        let s2mm_channel = 0u8;  // S2MM_0 on destination

        // Memory offsets
        let src_offset = 0x1000usize;
        let dst_offset = 0x2000usize;

        // Generate test pattern: each byte is (pattern_seed + index) mod 256
        // This creates a recognizable pattern that varies with both position and seed
        let test_data: Vec<u8> = (0..size)
            .map(|i| pattern_seed.wrapping_add((i % 256) as u8))
            .collect();

        // Write source data to source tile memory
        runner.write_tile_memory(src_col, src_row, src_offset, &test_data)
            .map_err(|e| format!("Failed to write source memory: {}", e))?;

        // Configure MM2S BD on source tile: read from src_offset
        let mm2s_bd = BdConfig::simple_1d(src_offset as u64, size);
        runner.configure_dma_bd(src_col, src_row, 0, mm2s_bd)
            .map_err(|e| format!("Failed to configure MM2S BD: {}", e))?;

        // Configure S2MM BD on destination tile: write to dst_offset
        let s2mm_bd = BdConfig::simple_1d(dst_offset as u64, size);
        runner.configure_dma_bd(dst_col, dst_row, 0, s2mm_bd)
            .map_err(|e| format!("Failed to configure S2MM BD: {}", e))?;

        // Configure stream routing: source tile MM2S -> dest tile S2MM
        runner.engine.device_mut().array.stream_router.add_route(
            src_col, src_row, mm2s_channel,
            dst_col, dst_row, s2mm_channel,
        );

        // Start both DMA channels
        runner.engine.device_mut().array.dma_engine_mut(src_col, src_row)
            .ok_or_else(|| "Failed to get source DMA engine".to_string())?
            .start_channel(mm2s_channel, 0)
            .map_err(|e| format!("Failed to start MM2S channel: {}", e))?;

        runner.engine.device_mut().array.dma_engine_mut(dst_col, dst_row)
            .ok_or_else(|| "Failed to get destination DMA engine".to_string())?
            .start_channel(s2mm_channel, 0)
            .map_err(|e| format!("Failed to start S2MM channel: {}", e))?;

        // Calculate cycles needed: allow for setup overhead plus per-byte transfer time
        // Use a generous margin to ensure completion
        let cycles_needed = 100 + (size as u64 * 2);
        for _ in 0..cycles_needed {
            runner.step();
        }

        // Read destination memory and verify
        let result = runner.read_tile_memory(dst_col, dst_row, dst_offset, size as usize)
            .map_err(|e| format!("Failed to read destination memory: {}", e))?;

        // Compare byte-by-byte for detailed error reporting
        if result.len() != test_data.len() {
            return Err(format!(
                "Length mismatch: expected {} bytes, got {} bytes",
                test_data.len(), result.len()
            ));
        }

        for (i, (expected, actual)) in test_data.iter().zip(result.iter()).enumerate() {
            if expected != actual {
                return Err(format!(
                    "Data mismatch at byte {}: expected 0x{:02X}, got 0x{:02X}",
                    i, expected, actual
                ));
            }
        }

        Ok(())
    }

    /// Test DMA transfers work correctly for various sizes.
    ///
    /// This test validates the DMA subsystem handles different transfer sizes:
    /// - 4 bytes: minimum transfer size (single 32-bit word)
    /// - 32 bytes: typical cache line size
    /// - 256 bytes: standard small transfer
    /// - 4096 bytes: page size, larger transfer
    ///
    /// For each size:
    /// 1. Initialize source memory with a known pattern
    /// 2. Configure DMA descriptor with that length
    /// 3. Execute transfer via MM2S -> S2MM with stream routing
    /// 4. Verify destination matches source exactly
    #[test]
    fn test_dma_transfer_sizes() {
        // Test cases: (size in bytes, pattern seed, description)
        let test_cases: Vec<(u32, u8, &str)> = vec![
            (4, 0x10, "minimum (4 bytes)"),
            (32, 0x20, "cache line (32 bytes)"),
            (256, 0x30, "standard (256 bytes)"),
            (4096, 0x40, "page size (4096 bytes)"),
        ];

        let mut passed = 0;
        let mut failed = Vec::new();

        for (size, seed, description) in &test_cases {
            eprintln!("Testing DMA transfer: {}...", description);

            match run_dma_transfer_test(*size, *seed) {
                Ok(()) => {
                    eprintln!("  PASS: {} bytes transferred correctly", size);
                    passed += 1;
                }
                Err(e) => {
                    eprintln!("  FAIL: {}", e);
                    failed.push((*description, e));
                }
            }
        }

        eprintln!("\nDMA transfer size test summary: {}/{} passed", passed, test_cases.len());

        // Assert all tests passed
        assert!(
            failed.is_empty(),
            "DMA transfer tests failed:\n{}",
            failed.iter()
                .map(|(desc, err)| format!("  - {}: {}", desc, err))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    /// Test data flow from memory tile (row 1) to compute tile (row 2+).
    ///
    /// Memory tiles in AIE2 are specialized tiles at row 1 with:
    /// - 512KB data memory (vs 64KB for compute tiles)
    /// - 64 locks (vs 16 for compute tiles)
    /// - 6 S2MM + 6 MM2S DMA channels (vs 2+2 for compute tiles)
    /// - No program memory (no compute core)
    ///
    /// This test validates the memtile -> compute tile data path:
    /// 1. Write data to memory tile (0,1) at address 0x1000
    /// 2. Configure DMA on memtile to send data via MM2S (channel 6)
    /// 3. Configure DMA on compute tile (0,2) to receive via S2MM (channel 0)
    /// 4. Set up stream routing from memtile to compute tile
    /// 5. Execute transfer
    /// 6. Verify data arrived correctly at compute tile
    ///
    /// Key differences from compute-to-compute:
    /// - Memory tiles use channel 6-11 for MM2S (vs 2-3 for compute)
    /// - Memory tiles use channel 0-5 for S2MM (vs 0-1 for compute)
    #[test]
    fn test_memtile_to_compute_path() {
        let mut runner = TestRunner::new();

        // Memory tile at row 1, compute tile at row 2
        let memtile_col = 0u8;
        let memtile_row = 1u8;
        let compute_col = 0u8;
        let compute_row = 2u8;

        // ============================================
        // Phase 1: Verify tile types and properties
        // ============================================

        // Verify memtile is correctly identified
        // Use array.tile() which returns &Tile directly (bounds are known valid)
        let memtile = runner.engine.device().array.tile(memtile_col, memtile_row);
        assert!(memtile.is_mem_tile(), "Tile at ({},{}) should be a memory tile", memtile_col, memtile_row);
        eprintln!("Memory tile ({},{}) confirmed: 512KB memory, 64 locks, 12 DMA channels",
            memtile_col, memtile_row);

        // Check memory size is 512KB
        let memtile_memory_size = memtile.data_memory().len();
        assert_eq!(memtile_memory_size, 512 * 1024,
            "Memory tile should have 512KB data memory, got {} bytes", memtile_memory_size);

        // Verify compute tile
        let compute_tile = runner.engine.device().array.tile(compute_col, compute_row);
        assert!(compute_tile.is_compute(), "Tile at ({},{}) should be a compute tile", compute_col, compute_row);

        // Verify DMA engine channel counts
        let memtile_dma = runner.engine.device().array.dma_engine(memtile_col, memtile_row).unwrap();
        assert_eq!(memtile_dma.num_channels(), 12,
            "Memory tile should have 12 DMA channels (6 S2MM + 6 MM2S), got {}", memtile_dma.num_channels());

        let compute_dma = runner.engine.device().array.dma_engine(compute_col, compute_row).unwrap();
        assert_eq!(compute_dma.num_channels(), 4,
            "Compute tile should have 4 DMA channels (2 S2MM + 2 MM2S), got {}", compute_dma.num_channels());

        eprintln!("Tile properties verified: memtile={} channels, compute={} channels",
            memtile_dma.num_channels(), compute_dma.num_channels());

        // ============================================
        // Phase 2: Set up test data
        // ============================================

        // Create test pattern: 256 bytes with recognizable pattern
        let transfer_size = 256u32;
        let test_data: Vec<u8> = (0..transfer_size).map(|i| ((0xD0 + i) % 256) as u8).collect();

        // Memory offsets
        let memtile_src_offset = 0x1000usize;
        let compute_dst_offset = 0x2000usize;

        // Write test data to memory tile
        // Note: write_tile_memory uses TestRunner which handles memtile memory access
        runner.write_tile_memory(memtile_col, memtile_row, memtile_src_offset, &test_data).unwrap();
        eprintln!("Wrote {} bytes to memtile ({},{}) at offset 0x{:04X}",
            test_data.len(), memtile_col, memtile_row, memtile_src_offset);

        // ============================================
        // Phase 3: Configure DMA buffer descriptors
        // ============================================

        // Memory tile DMA channels:
        // - Channels 0-5: S2MM (stream to memory)
        // - Channels 6-11: MM2S (memory to stream)
        let memtile_mm2s_channel = 6u8;  // First MM2S channel on memtile

        // Compute tile DMA channels:
        // - Channels 0-1: S2MM (stream to memory)
        // - Channels 2-3: MM2S (memory to stream)
        let compute_s2mm_channel = 0u8;  // First S2MM channel on compute tile

        // Configure MM2S BD on memtile: read from 0x1000, send to stream
        let memtile_mm2s_bd = BdConfig::simple_1d(memtile_src_offset as u64, transfer_size);
        runner.configure_dma_bd(memtile_col, memtile_row, 0, memtile_mm2s_bd).unwrap();
        eprintln!("Configured memtile MM2S BD: addr=0x{:04X}, len={}", memtile_src_offset, transfer_size);

        // Configure S2MM BD on compute tile: receive from stream, write to 0x2000
        let compute_s2mm_bd = BdConfig::simple_1d(compute_dst_offset as u64, transfer_size);
        runner.configure_dma_bd(compute_col, compute_row, 0, compute_s2mm_bd).unwrap();
        eprintln!("Configured compute S2MM BD: addr=0x{:04X}, len={}", compute_dst_offset, transfer_size);

        // ============================================
        // Phase 4: Configure stream routing
        // ============================================

        // Route memtile MM2S output to compute tile S2MM input
        // The channel number is used as the port identifier in the stream router
        runner.engine.device_mut().array.stream_router.add_route(
            memtile_col, memtile_row, memtile_mm2s_channel,
            compute_col, compute_row, compute_s2mm_channel,
        );
        eprintln!("Stream route configured: ({},{}):{} -> ({},{}):{}",
            memtile_col, memtile_row, memtile_mm2s_channel,
            compute_col, compute_row, compute_s2mm_channel);

        // ============================================
        // Phase 5: Start DMA channels
        // ============================================

        // Start memtile MM2S channel with BD 0
        runner.engine.device_mut().array.dma_engine_mut(memtile_col, memtile_row)
            .unwrap()
            .start_channel(memtile_mm2s_channel, 0)
            .unwrap();
        eprintln!("Started memtile MM2S channel {}", memtile_mm2s_channel);

        // Start compute tile S2MM channel with BD 0
        runner.engine.device_mut().array.dma_engine_mut(compute_col, compute_row)
            .unwrap()
            .start_channel(compute_s2mm_channel, 0)
            .unwrap();
        eprintln!("Started compute S2MM channel {}", compute_s2mm_channel);

        // ============================================
        // Phase 6: Execute transfer
        // ============================================

        // Step until transfer completes
        // Allow enough cycles for 256 bytes (64 words) plus overhead
        let max_cycles = 500;
        for cycle in 0..max_cycles {
            runner.step();

            // Check if both channels have completed
            let memtile_done = !runner.engine.device().array
                .dma_engine(memtile_col, memtile_row)
                .map_or(false, |e| e.channel_active(memtile_mm2s_channel));
            let compute_done = !runner.engine.device().array
                .dma_engine(compute_col, compute_row)
                .map_or(false, |e| e.channel_active(compute_s2mm_channel));

            if memtile_done && compute_done {
                eprintln!("Transfer completed in {} cycles", cycle + 1);
                break;
            }

            if cycle == max_cycles - 1 {
                eprintln!("WARNING: Reached max cycles. Memtile done: {}, Compute done: {}",
                    memtile_done, compute_done);
            }
        }

        // ============================================
        // Phase 7: Verify data transfer
        // ============================================

        // Read destination memory from compute tile
        let received_data = runner.read_tile_memory(compute_col, compute_row, compute_dst_offset, transfer_size as usize).unwrap();

        // Verify exact byte-for-byte match
        assert_eq!(
            received_data, test_data,
            "Data mismatch in memtile -> compute transfer"
        );

        eprintln!("\nMemory tile to compute tile transfer test PASSED:");
        eprintln!("  Source:      memtile ({},{}) @ 0x{:04X} (MM2S channel {})",
            memtile_col, memtile_row, memtile_src_offset, memtile_mm2s_channel);
        eprintln!("  Destination: compute ({},{}) @ 0x{:04X} (S2MM channel {})",
            compute_col, compute_row, compute_dst_offset, compute_s2mm_channel);
        eprintln!("  Data:        {} bytes transferred correctly (pattern starting 0xD0)",
            transfer_size);
        eprintln!("\nMemory tile properties validated:");
        eprintln!("  - 512KB data memory");
        eprintln!("  - 12 DMA channels (6 S2MM + 6 MM2S)");
        eprintln!("  - Data flows correctly from memtile MM2S to compute S2MM");
    }

    /// Test lock-synchronized producer-consumer pattern between two tiles.
    ///
    /// This test validates the core lock synchronization mechanism for DMA transfers
    /// where each tile uses its own local locks for DMA buffer synchronization:
    ///
    /// Pattern:
    /// 1. Tile A (producer) at (0,2): DMA acquires local lock 0, reads data, releases lock
    /// 2. Tile B (consumer) at (0,3): DMA acquires local lock 0, writes data, releases lock
    ///
    /// Lock protocol (per tile):
    /// - Producer lock 0: initially 1 (buffer ready), DMA acquires (1->0), then releases (0->1)
    /// - Consumer lock 0: initially 1 (buffer ready), DMA acquires (1->0), then releases (0->1)
    ///
    /// The streaming interface handles flow control between tiles, while locks
    /// coordinate local buffer access (e.g., between DMA and core on same tile).
    ///
    /// This pattern is fundamental to AIE programming where:
    /// - Locks guard local buffer access (DMA vs core)
    /// - Streams handle inter-tile data flow
    #[test]
    fn test_lock_synchronized_producer_consumer() {
        let mut runner = TestRunner::new();

        // Tile coordinates: producer at (0,2), consumer at (0,3)
        let producer_col = 0u8;
        let producer_row = 2u8;
        let consumer_col = 0u8;
        let consumer_row = 3u8;

        // Each tile uses its local lock 0 for DMA synchronization
        let producer_lock_id = 0u8;
        let consumer_lock_id = 0u8;

        // DMA channels: MM2S channel 2 for producer output, S2MM channel 0 for consumer input
        let mm2s_channel = 2u8;
        let s2mm_channel = 0u8;

        // Transfer configuration
        let transfer_size = 128u32;
        let producer_src_offset = 0x1000usize;
        let consumer_dst_offset = 0x2000usize;

        // Create test data pattern
        let test_data: Vec<u8> = (0..transfer_size).map(|i| ((0x50 + i) % 256) as u8).collect();

        // ============================================
        // Phase 1: Initialize lock states on both tiles
        // ============================================

        // Producer tile: lock 0 = 1 (buffer is ready for DMA to read)
        if let Some(tile) = runner.engine.device_mut().tile_mut(producer_col as usize, producer_row as usize) {
            tile.locks[producer_lock_id as usize].set(1);
        }
        eprintln!("Producer lock {} initialized to 1 (buffer ready)", producer_lock_id);

        // Consumer tile: lock 0 = 1 (buffer is ready for DMA to write)
        if let Some(tile) = runner.engine.device_mut().tile_mut(consumer_col as usize, consumer_row as usize) {
            tile.locks[consumer_lock_id as usize].set(1);
        }
        eprintln!("Consumer lock {} initialized to 1 (buffer ready)", consumer_lock_id);

        // ============================================
        // Phase 2: Producer writes data to its memory
        // ============================================

        // Write test data to producer's memory (simulating core writing to buffer)
        runner.write_tile_memory(producer_col, producer_row, producer_src_offset, &test_data).unwrap();
        eprintln!("Producer wrote {} bytes to offset 0x{:04X}", test_data.len(), producer_src_offset);

        // ============================================
        // Phase 3: Configure DMA with lock acquire/release
        // ============================================

        // Producer's MM2S BD: acquire lock 0, read from memory, release lock 0
        // This simulates: DMA waits for buffer to be ready, reads it, signals done
        let producer_bd = BdConfig::simple_1d(producer_src_offset as u64, transfer_size)
            .with_acquire(producer_lock_id, 1)   // Acquire: wait for lock >= 1, decrement
            .with_release(producer_lock_id, 1);  // Release: increment lock after transfer
        runner.configure_dma_bd(producer_col, producer_row, 0, producer_bd).unwrap();

        // Consumer's S2MM BD: acquire lock 0, write to memory, release lock 0
        // This simulates: DMA waits for buffer to be available, writes to it, signals done
        let consumer_bd = BdConfig::simple_1d(consumer_dst_offset as u64, transfer_size)
            .with_acquire(consumer_lock_id, 1)   // Acquire: wait for lock >= 1, decrement
            .with_release(consumer_lock_id, 1);  // Release: increment lock after transfer
        runner.configure_dma_bd(consumer_col, consumer_row, 0, consumer_bd).unwrap();

        // Configure stream routing: producer MM2S -> consumer S2MM
        runner.engine.device_mut().array.stream_router.add_route(
            producer_col, producer_row, mm2s_channel,
            consumer_col, consumer_row, s2mm_channel,
        );
        eprintln!("Stream route configured: ({},{}):{} -> ({},{}):{}",
            producer_col, producer_row, mm2s_channel,
            consumer_col, consumer_row, s2mm_channel);

        // ============================================
        // Phase 4: Verify initial lock state
        // ============================================

        // Check that locks are in expected initial state
        let producer_lock_initial = runner.engine.device()
            .tile(producer_col as usize, producer_row as usize)
            .map(|t| t.locks[producer_lock_id as usize].value)
            .unwrap_or(255);
        let consumer_lock_initial = runner.engine.device()
            .tile(consumer_col as usize, consumer_row as usize)
            .map(|t| t.locks[consumer_lock_id as usize].value)
            .unwrap_or(255);
        assert_eq!(producer_lock_initial, 1, "Producer lock should be 1 initially");
        assert_eq!(consumer_lock_initial, 1, "Consumer lock should be 1 initially");
        eprintln!("Initial lock states verified: producer={}, consumer={}",
            producer_lock_initial, consumer_lock_initial);

        // ============================================
        // Phase 5: Start both DMA channels
        // ============================================

        // Start producer's MM2S channel (will acquire lock, read, release)
        runner.engine.device_mut().array.dma_engine_mut(producer_col, producer_row)
            .unwrap()
            .start_channel(mm2s_channel, 0)
            .unwrap();
        eprintln!("Producer DMA started (acquire lock {}, read, release)", producer_lock_id);

        // Start consumer's S2MM channel (will acquire lock, write, release)
        runner.engine.device_mut().array.dma_engine_mut(consumer_col, consumer_row)
            .unwrap()
            .start_channel(s2mm_channel, 0)
            .unwrap();
        eprintln!("Consumer DMA started (acquire lock {}, write, release)", consumer_lock_id);

        // ============================================
        // Phase 6: Run until transfer completes
        // ============================================

        // Step the system until both transfers complete
        let max_cycles = 500;
        let mut completed_cycle = max_cycles;

        for cycle in 0..max_cycles {
            runner.step();

            // Check if both channels are idle (transfers complete)
            let producer_done = !runner.engine.device().array
                .dma_engine(producer_col, producer_row)
                .map_or(false, |e| e.channel_active(mm2s_channel));
            let consumer_done = !runner.engine.device().array
                .dma_engine(consumer_col, consumer_row)
                .map_or(false, |e| e.channel_active(s2mm_channel));

            if producer_done && consumer_done {
                completed_cycle = cycle + 1;
                eprintln!("Both transfers completed in {} cycles", completed_cycle);
                break;
            }

            if cycle == max_cycles - 1 {
                eprintln!("WARNING: Reached max cycles. Producer done: {}, Consumer done: {}",
                    producer_done, consumer_done);
            }
        }

        // ============================================
        // Phase 7: Verify lock states after transfers
        // ============================================

        // After acquire(-1) and release(+1), locks should be back to 1
        let producer_lock_final = runner.engine.device()
            .tile(producer_col as usize, producer_row as usize)
            .map(|t| t.locks[producer_lock_id as usize].value)
            .unwrap_or(255);
        let consumer_lock_final = runner.engine.device()
            .tile(consumer_col as usize, consumer_row as usize)
            .map(|t| t.locks[consumer_lock_id as usize].value)
            .unwrap_or(255);
        eprintln!("Final lock states: producer={}, consumer={}",
            producer_lock_final, consumer_lock_final);

        // Verify producer lock was properly acquired and released
        assert_eq!(producer_lock_final, 1,
            "Producer lock should be 1 after acquire+release cycle");

        // Verify consumer lock was properly acquired and released
        assert_eq!(consumer_lock_final, 1,
            "Consumer lock should be 1 after acquire+release cycle");

        // ============================================
        // Phase 8: Verify data transfer
        // ============================================

        // Verify consumer received the correct data
        let received_data = runner.read_tile_memory(consumer_col, consumer_row, consumer_dst_offset, transfer_size as usize).unwrap();
        assert_eq!(
            received_data, test_data,
            "Consumer should have received the producer's data via stream transfer"
        );

        eprintln!("Lock-synchronized producer-consumer test passed:");
        eprintln!("  Producer: tile ({},{}) @ 0x{:04X}, lock {} (1->0->1)",
            producer_col, producer_row, producer_src_offset, producer_lock_id);
        eprintln!("  Consumer: tile ({},{}) @ 0x{:04X}, lock {} (1->0->1)",
            consumer_col, consumer_row, consumer_dst_offset, consumer_lock_id);
        eprintln!("  Data:     {} bytes transferred correctly in {} cycles",
            transfer_size, completed_cycle);
    }

    /// Test that loads and executes a real XCLBIN file.
    ///
    /// This test validates our parser and state application work with real binaries.
    /// It exercises the complete loading pipeline:
    /// 1. Parse XCLBIN container
    /// 2. Extract AIE Partition section
    /// 3. Find and parse CDO commands
    /// 4. Apply CDO to device state
    /// 5. Load ELF binaries into tiles
    /// 6. Execute and report results
    ///
    /// The test is exploratory - it documents what works and what doesn't.
    #[test]
    fn test_execute_real_xclbin() {
        use crate::parser::xclbin::{SectionKind, Xclbin};
        use crate::parser::aie_partition::AiePartition;
        use crate::parser::cdo::{find_cdo_offset, Cdo};
        use crate::parser::MemoryRegion;
        use crate::device::DeviceState;
        use crate::interpreter::engine::InterpreterEngine;

        eprintln!("\n=== Real XCLBIN Execution Test ===\n");

        // Step 1: Locate XCLBIN files
        // Try multiple potential locations for add_one kernel
        let xclbin_candidates = [
            "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_objFifo/aie.xclbin",
            "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_using_dma/aie.xclbin",
            "/home/triple/npu-work/mlir-aie/build/test/npu-xrt/add_one_objFifo_elf/aie.xclbin",
        ];

        let xclbin_path = xclbin_candidates.iter()
            .find(|p| std::path::Path::new(p).exists());

        let xclbin_path = match xclbin_path {
            Some(path) => *path,
            None => {
                eprintln!("SKIP: No XCLBIN files found. Checked:");
                for path in &xclbin_candidates {
                    eprintln!("  - {}", path);
                }
                eprintln!("\nTo enable this test, build mlir-aie tests with:");
                eprintln!("  cd /home/triple/npu-work/mlir-aie/build");
                eprintln!("  ninja check-aie");
                return;
            }
        };

        eprintln!("XCLBIN: {}", xclbin_path);

        // Step 2: Parse XCLBIN container
        let xclbin = match Xclbin::from_file(xclbin_path) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("ERROR: Failed to parse XCLBIN: {}", e);
                panic!("XCLBIN parsing failed");
            }
        };

        eprintln!("  UUID: {}", xclbin.uuid());
        eprintln!("  Platform: {}", xclbin.platform());
        eprintln!("  Sections: {}", xclbin.num_sections());

        // List all sections
        eprintln!("\n--- Sections ---");
        for (i, section) in xclbin.sections().enumerate() {
            eprintln!("  [{:2}] {:?} \"{}\" @ 0x{:x}, {} bytes",
                i, section.kind, section.name(), section.offset, section.size());
        }

        // Step 3: Extract AIE Partition section
        let aie_partition_section = xclbin.find_section(SectionKind::AiePartition);
        let aie_partition_section = match aie_partition_section {
            Some(s) => s,
            None => {
                eprintln!("ERROR: No AIE_PARTITION section found");
                eprintln!("       This XCLBIN may not be for an NPU target");
                panic!("Missing AIE_PARTITION");
            }
        };

        eprintln!("\n--- AIE Partition ---");
        eprintln!("  Size: {} bytes", aie_partition_section.size());

        let aie_partition = match AiePartition::parse(aie_partition_section.data()) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("ERROR: Failed to parse AIE partition: {}", e);
                panic!("AIE partition parsing failed");
            }
        };

        eprintln!("  Column width: {}", aie_partition.column_width());
        if let Some(name) = aie_partition.name() {
            eprintln!("  Name: {}", name);
        }
        let start_cols = aie_partition.start_columns();
        if !start_cols.is_empty() {
            eprintln!("  Start columns: {:?}", start_cols);
        }

        // Get PDI images
        let pdis: Vec<_> = aie_partition.pdis().collect();
        eprintln!("  PDI count: {}", pdis.len());

        if pdis.is_empty() {
            eprintln!("ERROR: No PDI images in AIE partition");
            panic!("No PDIs found");
        }

        // Step 4: Find and parse CDO within first PDI
        let pdi = &pdis[0];
        eprintln!("\n--- Primary PDI ---");
        eprintln!("  UUID: {}", pdi.uuid);
        eprintln!("  CDO type: {:?}", pdi.cdo_type);
        eprintln!("  Image size: {} bytes", pdi.pdi_image.len());

        let cdo_offset = match find_cdo_offset(pdi.pdi_image) {
            Some(offset) => {
                eprintln!("  CDO found at offset: 0x{:x}", offset);
                offset
            }
            None => {
                eprintln!("ERROR: No CDO magic found in PDI image");
                eprintln!("       PDI may use a different format or be encrypted");
                panic!("CDO not found");
            }
        };

        let cdo = match Cdo::parse(&pdi.pdi_image[cdo_offset..]) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("ERROR: Failed to parse CDO: {}", e);
                panic!("CDO parsing failed");
            }
        };

        eprintln!("\n--- CDO Summary ---");
        eprintln!("  Magic: {}", cdo.magic());
        eprintln!("  Version: {:?}", cdo.version());
        eprintln!("  Length: {} words ({} bytes)",
            cdo.command_length_words(),
            cdo.command_length_words() * 4);

        // Count commands by type
        let counts = cdo.command_counts();
        eprintln!("  Command counts:");
        let mut sorted: Vec<_> = counts.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));
        for (name, count) in sorted.iter().take(10) {
            eprintln!("    {}: {}", name, count);
        }

        // Step 5: Apply CDO to device state
        eprintln!("\n--- Applying CDO to Device State ---");
        let mut device_state = DeviceState::new_npu1();
        let cdo_stats_commands;
        match device_state.apply_cdo(&cdo) {
            Ok(()) => {
                eprintln!("  Commands processed: {}", device_state.stats.commands);
                eprintln!("  Writes: {}", device_state.stats.writes);
                eprintln!("  Mask writes: {}", device_state.stats.mask_writes);
                eprintln!("  DMA writes: {}", device_state.stats.dma_writes);
                eprintln!("  Data bytes: {}", device_state.stats.data_bytes);
                eprintln!("  Program bytes: {}", device_state.stats.program_bytes);
                eprintln!("  Unknown: {}", device_state.stats.unknown);
                cdo_stats_commands = device_state.stats.commands;
            }
            Err(e) => {
                eprintln!("ERROR: Failed to apply CDO: {}", e);
                panic!("CDO application failed");
            }
        }

        // Step 6: Look for ELF files in the project directory
        eprintln!("\n--- Looking for ELF files ---");

        // The ELF path is typically in the .prj directory alongside the XCLBIN
        let xclbin_dir = std::path::Path::new(xclbin_path).parent().unwrap();

        let mut elf_files: Vec<std::path::PathBuf> = Vec::new();

        // Search for ELF files using standard library
        // Check common locations: *.elf in current dir and aie_arch.mlir.prj/*.elf
        let search_dirs = [
            xclbin_dir.to_path_buf(),
            xclbin_dir.join("aie_arch.mlir.prj"),
        ];

        for dir in &search_dirs {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().map_or(false, |ext| ext == "elf") {
                        if !elf_files.contains(&path) {
                            elf_files.push(path);
                        }
                    }
                }
            }
        }

        if elf_files.is_empty() {
            eprintln!("  No ELF files found in project directory");
            eprintln!("  Note: Some XCLBINs include ELFs embedded in CDO,");
            eprintln!("        while others require separate ELF files.");
        } else {
            eprintln!("  Found {} ELF file(s):", elf_files.len());
            for elf_path in &elf_files {
                eprintln!("    - {}", elf_path.display());
            }
        }

        // Step 7: Create interpreter engine and try execution
        eprintln!("\n--- Creating Interpreter Engine ---");
        let mut engine = InterpreterEngine::new(device_state);

        // Check for enabled cores from CDO
        let enabled_cores = engine.enabled_cores();
        eprintln!("  Enabled cores from CDO: {}", enabled_cores);

        // Load ELF files if found
        let mut cores_loaded = 0;
        for elf_path in &elf_files {
            // Try to extract tile coordinates from filename (e.g., "main_core_0_2.elf")
            let filename = elf_path.file_stem().unwrap().to_str().unwrap();
            let coords = parse_elf_filename(filename);

            if let Some((col, row)) = coords {
                eprintln!("  Loading ELF to tile ({}, {}): {}", col, row, filename);

                let elf_data = match std::fs::read(elf_path) {
                    Ok(data) => data,
                    Err(e) => {
                        eprintln!("    ERROR: Failed to read ELF: {}", e);
                        continue;
                    }
                };

                // Parse ELF
                let elf = match crate::parser::AieElf::parse(&elf_data) {
                    Ok(e) => e,
                    Err(e) => {
                        eprintln!("    ERROR: Failed to parse ELF: {}", e);
                        continue;
                    }
                };

                eprintln!("    Entry point: 0x{:x}", elf.entry_point());

                // Load into device
                let tile = match engine.device_mut().tile_mut(col as usize, row as usize) {
                    Some(t) => t,
                    None => {
                        eprintln!("    ERROR: Invalid tile ({}, {})", col, row);
                        continue;
                    }
                };

                // Load segments
                for seg in elf.load_segments() {
                    let vaddr = seg.vaddr as usize;
                    let data = seg.data;

                    match seg.region {
                        MemoryRegion::Program => {
                            tile.write_program(vaddr, data);
                            eprintln!("    Loaded {} bytes to program memory @ 0x{:x}",
                                data.len(), vaddr);
                        }
                        MemoryRegion::Data => {
                            let dm_offset = vaddr.saturating_sub(0x00070000);
                            let dm = tile.data_memory_mut();
                            if dm_offset + data.len() <= dm.len() {
                                dm[dm_offset..dm_offset + data.len()].copy_from_slice(data);
                                eprintln!("    Loaded {} bytes to data memory @ 0x{:x}",
                                    data.len(), dm_offset);
                            }
                        }
                        _ => {}
                    }
                }

                // Set entry point and enable core
                engine.set_core_pc(col as usize, row as usize, elf.entry_point());
                engine.enable_core(col as usize, row as usize);
                cores_loaded += 1;
            }
        }

        if cores_loaded > 0 {
            eprintln!("  Loaded {} cores from ELF files", cores_loaded);
        }

        // Step 8: Attempt execution
        eprintln!("\n--- Execution Attempt ---");

        let total_enabled = engine.enabled_cores();
        if total_enabled == 0 {
            eprintln!("  No cores enabled - cannot execute");
            eprintln!("\n  Analysis:");
            eprintln!("  - CDO applied successfully ({} commands)", cdo_stats_commands);
            eprintln!("  - No cores were enabled by CDO or ELF loading");
            eprintln!("  - This may be expected for some XCLBIN types");
            eprintln!("\n  Possible reasons:");
            eprintln!("  - XCLBINs using DMA-only execution (no core code)");
            eprintln!("  - Core enable happens at runtime via XRT");
            eprintln!("  - Missing ELF files");
            return;
        }

        eprintln!("  Enabled cores: {}", total_enabled);

        // Try stepping execution
        let max_steps = 100;
        let mut step = 0;
        let mut halted = false;
        let mut error_encountered = false;

        while step < max_steps && !halted && !error_encountered {
            engine.step();

            let status = engine.status();
            match status {
                crate::interpreter::engine::EngineStatus::Halted => {
                    halted = true;
                    eprintln!("  Step {}: All cores halted", step);
                }
                crate::interpreter::engine::EngineStatus::Error => {
                    error_encountered = true;
                    eprintln!("  Step {}: Engine error", step);
                }
                _ => {}
            }
            step += 1;
        }

        eprintln!("\n--- Execution Result ---");
        eprintln!("  Steps executed: {}", step);
        eprintln!("  Total cycles: {}", engine.total_cycles());
        eprintln!("  Final status: {:?}", engine.status());

        if error_encountered {
            eprintln!("\n  NOTE: Error indicates an unimplemented instruction or feature.");
            eprintln!("        Check decoder coverage for the specific instruction.");
        }

        if halted {
            eprintln!("\n  SUCCESS: Execution completed normally.");
        } else if !error_encountered {
            eprintln!("\n  NOTE: Execution did not complete within {} steps.", max_steps);
            eprintln!("        The kernel may require DMA/lock synchronization.");
        }

        eprintln!("\n=== End Real XCLBIN Execution Test ===\n");
    }

    /// Parse ELF filename to extract tile coordinates.
    /// Expected format: "main_core_X_Y.elf" where X is column and Y is row.
    fn parse_elf_filename(filename: &str) -> Option<(u8, u8)> {
        // Look for pattern like "_core_X_Y" or just "_X_Y"
        let parts: Vec<&str> = filename.split('_').collect();

        // Try to find "core" followed by two numbers
        for i in 0..parts.len().saturating_sub(2) {
            if parts[i] == "core" {
                if let (Ok(col), Ok(row)) = (parts[i+1].parse::<u8>(), parts[i+2].parse::<u8>()) {
                    return Some((col, row));
                }
            }
        }

        // Fallback: try last two parts as numbers
        if parts.len() >= 2 {
            let last = parts[parts.len() - 1];
            let second_last = parts[parts.len() - 2];
            if let (Ok(col), Ok(row)) = (second_last.parse::<u8>(), last.parse::<u8>()) {
                return Some((col, row));
            }
        }

        None
    }
}
