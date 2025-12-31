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
}
