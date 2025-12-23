//! Memory unit execution.
//!
//! Handles load/store operations between registers and tile memory.
//!
//! # Addressing
//!
//! AIE2 uses pointer registers (p0-p7) for addressing with optional
//! post-modify (add immediate or modifier register after access).
//!
//! # Memory Layout
//!
//! - **Data Memory**: 64KB at 0x00000-0x0FFFF (compute tile)
//! - **Program Memory**: 64KB at 0x20000-0x2FFFF (read-only)

use crate::device::tile::Tile;
use crate::interpreter::bundle::{MemWidth, Operation, Operand, PostModify, SlotOp};
use crate::interpreter::state::ExecutionContext;

/// Memory unit for load/store operations.
pub struct MemoryUnit;

impl MemoryUnit {
    /// Execute a memory operation.
    ///
    /// Returns `true` if the operation was handled, `false` if not a memory op.
    pub fn execute(op: &SlotOp, ctx: &mut ExecutionContext, tile: &mut Tile) -> bool {
        match &op.op {
            Operation::Load { width, post_modify } => {
                Self::execute_load(op, ctx, tile, *width, post_modify);
                true
            }

            Operation::Store { width, post_modify } => {
                Self::execute_store(op, ctx, tile, *width, post_modify);
                true
            }

            _ => false, // Not a memory operation
        }
    }

    /// Execute a load operation.
    fn execute_load(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        tile: &Tile,
        width: MemWidth,
        post_modify: &PostModify,
    ) {
        // Get address from source operand
        let addr = Self::get_address(op, ctx);

        // Read from memory
        let value = Self::read_memory(tile, addr, width);

        // Write to destination register
        Self::write_dest(op, ctx, value, width);

        // Apply post-modify to address register
        Self::apply_post_modify(op, ctx, post_modify);
    }

    /// Execute a store operation.
    fn execute_store(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        tile: &mut Tile,
        width: MemWidth,
        post_modify: &PostModify,
    ) {
        // Get address from first source operand
        let addr = Self::get_address(op, ctx);

        // Get value from second source (or dest for store) operand
        let value = Self::get_store_value(op, ctx, width);

        // Write to memory
        Self::write_memory(tile, addr, value, width);

        // Apply post-modify to address register
        Self::apply_post_modify(op, ctx, post_modify);
    }

    /// Get address from memory operand or pointer register.
    fn get_address(op: &SlotOp, ctx: &ExecutionContext) -> u32 {
        op.sources.first().map_or(0, |src| match src {
            Operand::Memory { base, offset } => {
                let base_addr = ctx.pointer.read(*base);
                base_addr.wrapping_add(*offset as i32 as u32)
            }
            Operand::PointerReg(r) => ctx.pointer.read(*r),
            Operand::ScalarReg(r) => ctx.scalar.read(*r),
            Operand::Immediate(v) => *v as u32,
            _ => 0,
        })
    }

    /// Get value to store.
    fn get_store_value(op: &SlotOp, ctx: &ExecutionContext, width: MemWidth) -> u64 {
        // For stores, the value is typically in the second source or in dest
        let operand = op.sources.get(1).or(op.dest.as_ref());

        operand.map_or(0, |src| match src {
            Operand::ScalarReg(r) => ctx.scalar.read(*r) as u64,
            Operand::VectorReg(r) => {
                // For vector stores, pack into u64 (first 2 lanes)
                let vec = ctx.vector.read(*r);
                if width == MemWidth::Vector256 {
                    // Full vector - return first 64 bits for now
                    ((vec[1] as u64) << 32) | (vec[0] as u64)
                } else {
                    vec[0] as u64
                }
            }
            Operand::Immediate(v) => *v as u64,
            _ => 0,
        })
    }

    /// Read from tile memory.
    fn read_memory(tile: &Tile, addr: u32, width: MemWidth) -> u64 {
        let addr = addr as usize;
        let mem = tile.data_memory();

        if addr >= mem.len() {
            return 0;
        }

        match width {
            MemWidth::Byte => {
                if addr < mem.len() {
                    mem[addr] as u64
                } else {
                    0
                }
            }
            MemWidth::HalfWord => {
                if addr + 1 < mem.len() {
                    u16::from_le_bytes([mem[addr], mem[addr + 1]]) as u64
                } else {
                    0
                }
            }
            MemWidth::Word => {
                if addr + 3 < mem.len() {
                    u32::from_le_bytes([mem[addr], mem[addr + 1], mem[addr + 2], mem[addr + 3]])
                        as u64
                } else {
                    0
                }
            }
            MemWidth::DoubleWord => {
                if addr + 7 < mem.len() {
                    u64::from_le_bytes([
                        mem[addr],
                        mem[addr + 1],
                        mem[addr + 2],
                        mem[addr + 3],
                        mem[addr + 4],
                        mem[addr + 5],
                        mem[addr + 6],
                        mem[addr + 7],
                    ])
                } else {
                    0
                }
            }
            MemWidth::QuadWord | MemWidth::Vector256 => {
                // Return first 64 bits for scalar representation
                if addr + 7 < mem.len() {
                    u64::from_le_bytes([
                        mem[addr],
                        mem[addr + 1],
                        mem[addr + 2],
                        mem[addr + 3],
                        mem[addr + 4],
                        mem[addr + 5],
                        mem[addr + 6],
                        mem[addr + 7],
                    ])
                } else {
                    0
                }
            }
        }
    }

    /// Write to tile memory.
    fn write_memory(tile: &mut Tile, addr: u32, value: u64, width: MemWidth) {
        let addr = addr as usize;
        let mem = tile.data_memory_mut();

        if addr >= mem.len() {
            return;
        }

        match width {
            MemWidth::Byte => {
                if addr < mem.len() {
                    mem[addr] = value as u8;
                }
            }
            MemWidth::HalfWord => {
                if addr + 1 < mem.len() {
                    let bytes = (value as u16).to_le_bytes();
                    mem[addr] = bytes[0];
                    mem[addr + 1] = bytes[1];
                }
            }
            MemWidth::Word => {
                if addr + 3 < mem.len() {
                    let bytes = (value as u32).to_le_bytes();
                    mem[addr..addr + 4].copy_from_slice(&bytes);
                }
            }
            MemWidth::DoubleWord => {
                if addr + 7 < mem.len() {
                    let bytes = value.to_le_bytes();
                    mem[addr..addr + 8].copy_from_slice(&bytes);
                }
            }
            MemWidth::QuadWord => {
                if addr + 15 < mem.len() {
                    let bytes = value.to_le_bytes();
                    mem[addr..addr + 8].copy_from_slice(&bytes);
                    // Upper 64 bits are zero
                    mem[addr + 8..addr + 16].fill(0);
                }
            }
            MemWidth::Vector256 => {
                if addr + 31 < mem.len() {
                    let bytes = value.to_le_bytes();
                    mem[addr..addr + 8].copy_from_slice(&bytes);
                    // Upper bytes are zero (would need full vector context)
                    mem[addr + 8..addr + 32].fill(0);
                }
            }
        }
    }

    /// Write loaded value to destination register.
    fn write_dest(op: &SlotOp, ctx: &mut ExecutionContext, value: u64, width: MemWidth) {
        if let Some(dest) = &op.dest {
            match dest {
                Operand::ScalarReg(r) => {
                    ctx.scalar.write(*r, value as u32);
                }
                Operand::VectorReg(r) => {
                    if width == MemWidth::Vector256 {
                        // Would need to read full 256 bits from memory
                        // For now, put value in first two lanes
                        let mut vec = [0u32; 8];
                        vec[0] = value as u32;
                        vec[1] = (value >> 32) as u32;
                        ctx.vector.write(*r, vec);
                    } else {
                        // Scalar value into lane 0
                        let mut vec = [0u32; 8];
                        vec[0] = value as u32;
                        ctx.vector.write(*r, vec);
                    }
                }
                _ => {}
            }
        }
    }

    /// Apply post-modify to the base address register.
    fn apply_post_modify(op: &SlotOp, ctx: &mut ExecutionContext, post_modify: &PostModify) {
        // Find the pointer register used as base
        let ptr_reg = op.sources.first().and_then(|src| match src {
            Operand::Memory { base, .. } => Some(*base),
            Operand::PointerReg(r) => Some(*r),
            _ => None,
        });

        if let Some(reg) = ptr_reg {
            match post_modify {
                PostModify::None => {}
                PostModify::Immediate(imm) => {
                    ctx.pointer.add(reg, *imm as i32);
                }
                PostModify::Register(m) => {
                    let modifier = ctx.modifier.read_signed(*m);
                    ctx.pointer.add(reg, modifier);
                }
            }
        }
    }

    /// Read a vector from memory (256 bits = 32 bytes).
    pub fn read_vector_from_memory(tile: &Tile, addr: u32) -> [u32; 8] {
        let addr = addr as usize;
        let mem = tile.data_memory();
        let mut result = [0u32; 8];

        if addr + 31 < mem.len() {
            for i in 0..8 {
                let offset = addr + i * 4;
                result[i] = u32::from_le_bytes([
                    mem[offset],
                    mem[offset + 1],
                    mem[offset + 2],
                    mem[offset + 3],
                ]);
            }
        }

        result
    }

    /// Write a vector to memory (256 bits = 32 bytes).
    pub fn write_vector_to_memory(tile: &mut Tile, addr: u32, value: [u32; 8]) {
        let addr = addr as usize;
        let mem = tile.data_memory_mut();

        if addr + 31 < mem.len() {
            for i in 0..8 {
                let offset = addr + i * 4;
                let bytes = value[i].to_le_bytes();
                mem[offset..offset + 4].copy_from_slice(&bytes);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpreter::bundle::SlotIndex;

    fn make_ctx() -> ExecutionContext {
        ExecutionContext::new()
    }

    fn make_tile() -> Tile {
        Tile::compute(0, 2)
    }

    #[test]
    fn test_load_word() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Write test data to memory
        tile.write_data_u32(0x100, 0xDEAD_BEEF);

        // p0 = 0x100
        ctx.pointer.write(0, 0x100);

        // r0 = [p0]
        let op = SlotOp::new(
            SlotIndex::Load,
            Operation::Load {
                width: MemWidth::Word,
                post_modify: PostModify::None,
            },
        )
        .with_dest(Operand::ScalarReg(0))
        .with_source(Operand::PointerReg(0));

        assert!(MemoryUnit::execute(&op, &mut ctx, &mut tile));
        assert_eq!(ctx.scalar.read(0), 0xDEAD_BEEF);
    }

    #[test]
    fn test_load_with_offset() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        tile.write_data_u32(0x108, 0xCAFE_BABE);
        ctx.pointer.write(0, 0x100);

        // r0 = [p0 + 8]
        let op = SlotOp::new(
            SlotIndex::Load,
            Operation::Load {
                width: MemWidth::Word,
                post_modify: PostModify::None,
            },
        )
        .with_dest(Operand::ScalarReg(0))
        .with_source(Operand::Memory { base: 0, offset: 8 });

        MemoryUnit::execute(&op, &mut ctx, &mut tile);
        assert_eq!(ctx.scalar.read(0), 0xCAFE_BABE);
    }

    #[test]
    fn test_load_with_post_modify_immediate() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        tile.write_data_u32(0x100, 0x1234_5678);
        ctx.pointer.write(0, 0x100);

        // r0 = [p0], p0 += 4
        let op = SlotOp::new(
            SlotIndex::Load,
            Operation::Load {
                width: MemWidth::Word,
                post_modify: PostModify::Immediate(4),
            },
        )
        .with_dest(Operand::ScalarReg(0))
        .with_source(Operand::PointerReg(0));

        MemoryUnit::execute(&op, &mut ctx, &mut tile);
        assert_eq!(ctx.scalar.read(0), 0x1234_5678);
        assert_eq!(ctx.pointer.read(0), 0x104); // Post-modified
    }

    #[test]
    fn test_load_with_post_modify_register() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        tile.write_data_u32(0x100, 0xABCD);
        ctx.pointer.write(0, 0x100);
        ctx.modifier.write(0, 0x10); // m0 = 16

        // r0 = [p0], p0 += m0
        let op = SlotOp::new(
            SlotIndex::Load,
            Operation::Load {
                width: MemWidth::Word,
                post_modify: PostModify::Register(0),
            },
        )
        .with_dest(Operand::ScalarReg(0))
        .with_source(Operand::PointerReg(0));

        MemoryUnit::execute(&op, &mut ctx, &mut tile);
        assert_eq!(ctx.scalar.read(0), 0xABCD);
        assert_eq!(ctx.pointer.read(0), 0x110);
    }

    #[test]
    fn test_store_word() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        ctx.pointer.write(0, 0x200);
        ctx.scalar.write(1, 0xFEED_FACE);

        // [p0] = r1
        let op = SlotOp::new(
            SlotIndex::Store,
            Operation::Store {
                width: MemWidth::Word,
                post_modify: PostModify::None,
            },
        )
        .with_dest(Operand::ScalarReg(1))
        .with_source(Operand::PointerReg(0));

        MemoryUnit::execute(&op, &mut ctx, &mut tile);
        assert_eq!(tile.read_data_u32(0x200), Some(0xFEED_FACE));
    }

    #[test]
    fn test_load_byte() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        tile.data_memory_mut()[0x50] = 0xAB;
        ctx.pointer.write(0, 0x50);

        let op = SlotOp::new(
            SlotIndex::Load,
            Operation::Load {
                width: MemWidth::Byte,
                post_modify: PostModify::None,
            },
        )
        .with_dest(Operand::ScalarReg(0))
        .with_source(Operand::PointerReg(0));

        MemoryUnit::execute(&op, &mut ctx, &mut tile);
        assert_eq!(ctx.scalar.read(0), 0xAB);
    }

    #[test]
    fn test_load_halfword() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        tile.data_memory_mut()[0x60] = 0xCD;
        tile.data_memory_mut()[0x61] = 0xAB;
        ctx.pointer.write(0, 0x60);

        let op = SlotOp::new(
            SlotIndex::Load,
            Operation::Load {
                width: MemWidth::HalfWord,
                post_modify: PostModify::None,
            },
        )
        .with_dest(Operand::ScalarReg(0))
        .with_source(Operand::PointerReg(0));

        MemoryUnit::execute(&op, &mut ctx, &mut tile);
        assert_eq!(ctx.scalar.read(0), 0xABCD);
    }

    #[test]
    fn test_vector_memory_helpers() {
        let mut tile = make_tile();

        let data = [1, 2, 3, 4, 5, 6, 7, 8];
        MemoryUnit::write_vector_to_memory(&mut tile, 0x300, data);

        let read_back = MemoryUnit::read_vector_from_memory(&tile, 0x300);
        assert_eq!(read_back, data);
    }
}
