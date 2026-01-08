//! Memory unit execution.
//!
//! Handles load/store operations between registers and tile memory.
//!
//! # Architecture Note
//!
//! Unlike [`ScalarAlu`](super::ScalarAlu) and [`VectorAlu`](super::VectorAlu),
//! the memory unit is NOT a legacy fallback - it handles actual memory access
//! that semantic dispatch cannot replicate (memory ops need tile access).
//!
//! ## Execution Flow
//!
//! ```text
//! CycleAccurateExecutor::execute_slot()
//!         |
//!         v
//!   execute_semantic(op, ctx)  <-- Pure register ops only
//!         |
//!         v
//!   ScalarAlu::execute(op, ctx)
//!         |
//!         v
//!   VectorAlu::execute(op, ctx)
//!         |
//!         v
//!   MemoryUnit::execute(op, ctx, tile)  <-- Memory access (this module)
//! ```
//!
//! Memory operations require tile access for actual reads/writes, so they
//! will always be handled here rather than in semantic dispatch.
//!
//! # Addressing
//!
//! AIE2 uses pointer registers (p0-p7) for addressing with optional
//! post-modify (add immediate or modifier register after access).
//!
//! The AGU (Address Generation Unit) generates 20-bit addresses spanning
//! 0x0000-0x3FFFF (256KB) across the 4 neighboring memory modules:
//! - 0x00000-0x0FFFF: South (local tile's own memory)
//! - 0x10000-0x1FFFF: West neighbor
//! - 0x20000-0x2FFFF: North neighbor
//! - 0x30000-0x3FFFF: East neighbor
//!
//! Linker-assigned addresses (like 0x70440) use higher bits to indicate
//! memory regions; these must be masked to the 18-bit AGU range.
//!
//! # Memory Layout
//!
//! - **Data Memory**: 64KB at 0x00000-0x0FFFF (compute tile)
//! - **Program Memory**: 16KB (read-only from core)

use crate::device::tile::Tile;
use crate::interpreter::bundle::{ElementType, MemWidth, Operation, Operand, PostModify, SlotOp};
use crate::interpreter::state::ExecutionContext;

/// Address mask for local tile memory (64KB).
/// Addresses from the AGU span 0x0000-0x3FFFF (256KB across 4 neighbors),
/// but local memory access only uses the lower 16 bits.
///
/// Note: Multi-tile memory access (addresses 0x10000-0x3FFFF) would need
/// routing to neighbor tiles - not yet implemented.
const LOCAL_MEMORY_MASK: u32 = 0xFFFF;

/// Memory unit for load/store operations.
pub struct MemoryUnit;

impl MemoryUnit {
    /// Execute a memory operation.
    ///
    /// Returns `true` if the operation was handled, `false` if not a memory op.
    pub fn execute(op: &SlotOp, ctx: &mut ExecutionContext, tile: &mut Tile) -> bool {
        #[cfg(test)]
        if matches!(&op.op, Operation::PointerMov) {
            eprintln!("[MEM EXEC] Got PointerMov: dest={:?} sources={:?}", op.dest, op.sources);
        }

        match &op.op {
            Operation::Load { width, post_modify } => {
                Self::execute_load(op, ctx, tile, *width, post_modify);
                true
            }

            Operation::Store { width, post_modify } => {
                Self::execute_store(op, ctx, tile, *width, post_modify);
                true
            }

            Operation::PointerAdd => {
                Self::execute_pointer_add(op, ctx);
                true
            }

            Operation::PointerMov => {
                Self::execute_pointer_mov(op, ctx);
                true
            }

            Operation::VectorLoadA { post_modify } => {
                Self::execute_vector_load_a(op, ctx, tile, post_modify);
                true
            }

            Operation::VectorLoadB { post_modify } => {
                Self::execute_vector_load_b(op, ctx, tile, post_modify);
                true
            }

            Operation::VectorLoadUnpack {
                from_type,
                to_type,
                post_modify,
            } => {
                Self::execute_vector_load_unpack(op, ctx, tile, *from_type, *to_type, post_modify);
                true
            }

            Operation::VectorStore { post_modify } => {
                Self::execute_vector_store(op, ctx, tile, post_modify);
                true
            }

            _ => false, // Not a memory operation
        }
    }

    /// Execute pointer add: ptr = ptr + offset.
    /// padda, paddb, padds instructions modify pointer registers for address generation.
    fn execute_pointer_add(op: &SlotOp, ctx: &mut ExecutionContext) {
        // Get destination pointer register
        let ptr_reg = match &op.dest {
            Some(Operand::PointerReg(r)) => *r,
            _ => return, // No valid destination
        };

        // Get current pointer value
        let ptr_value = ctx.pointer.read(ptr_reg);

        // Get offset from source operand
        let offset = match op.sources.first() {
            Some(Operand::Immediate(imm)) => *imm as i32,
            Some(Operand::ScalarReg(r)) => ctx.scalar.read(*r) as i32,
            Some(Operand::ModifierReg(r)) => ctx.modifier.read(*r) as i32,
            _ => 0,
        };

        // Add offset to pointer (wrapping)
        let new_value = (ptr_value as i32).wrapping_add(offset) as u32;

        #[cfg(test)]
        eprintln!("[PADD] p{}=0x{:04X} + {} = 0x{:04X} sources={:?} dest={:?}",
            ptr_reg, ptr_value, offset, new_value, op.sources, op.dest);

        ctx.pointer.write(ptr_reg, new_value);
    }

    /// Execute pointer move: ptr = value.
    /// mova, movb instructions set pointer registers.
    fn execute_pointer_mov(op: &SlotOp, ctx: &mut ExecutionContext) {
        // Get destination pointer register
        let ptr_reg = match &op.dest {
            Some(Operand::PointerReg(r)) => *r,
            _ => {
                #[cfg(test)]
                eprintln!("[PMOV] No dest pointer! dest={:?}", op.dest);
                return; // No valid destination
            }
        };

        // Get value from source operand
        let value = match op.sources.first() {
            Some(Operand::Immediate(imm)) => *imm as u32,
            Some(Operand::ScalarReg(r)) => ctx.scalar.read(*r),
            Some(Operand::PointerReg(r)) => ctx.pointer.read(*r),
            _ => 0,
        };

        #[cfg(test)]
        eprintln!("[PMOV] p{}=0x{:04X} sources={:?}", ptr_reg, value, op.sources);

        ctx.pointer.write(ptr_reg, value);
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

        log::debug!("[LOAD] addr=0x{:04X} sources={:?} dest={:?}", addr, op.sources, op.dest);

        // Handle full vector loads specially
        if width == MemWidth::Vector256 {
            let vec_data = Self::read_vector_from_memory(tile, addr);
            if let Some(Operand::VectorReg(r)) = &op.dest {
                ctx.vector.write(*r, vec_data);
            }
        } else {
            // Scalar/partial loads
            let value = Self::read_memory(tile, addr, width);
            log::debug!("[LOAD] loaded value=0x{:08X} ({}) to {:?}", value, value, op.dest);
            Self::write_dest(op, ctx, value, width);
        }

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
        // Get address using store-specific layout: sources[1]=ptr, sources[2]=offset
        let addr = Self::get_store_address(op, ctx);

        // Debug: log store operations with full pointer state
        log::debug!("[STORE] addr=0x{:04X} sources={:?} dest={:?} pointers=[p0=0x{:X},p1=0x{:X},p2=0x{:X},p3=0x{:X}]",
            addr, op.sources, op.dest,
            ctx.pointer.read(0), ctx.pointer.read(1), ctx.pointer.read(2), ctx.pointer.read(3));

        // Handle full vector stores specially
        if width == MemWidth::Vector256 {
            // Get vector register from source or dest
            let vec_reg = op
                .sources
                .get(1)
                .or(op.dest.as_ref())
                .and_then(|operand| match operand {
                    Operand::VectorReg(r) => Some(*r),
                    _ => None,
                });

            if let Some(r) = vec_reg {
                let vec_data = ctx.vector.read(r);
                Self::write_vector_to_memory(tile, addr, vec_data);
            }
        } else {
            // Scalar/partial stores
            let value = Self::get_store_value(op, ctx, width);
            log::debug!("[STORE] value=0x{:08X} width={:?}", value, width);
            Self::write_memory(tile, addr, value, width);
        }

        // Apply post-modify to address register
        Self::apply_post_modify(op, ctx, post_modify);
    }

    /// Execute vector load A (VLDA).
    ///
    /// Loads 256 bits (32 bytes) from memory into a vector register using the
    /// A-channel pointer (typically p0-p3). This is a dedicated wide load
    /// operation for vector data.
    fn execute_vector_load_a(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        tile: &Tile,
        post_modify: &PostModify,
    ) {
        let addr = Self::get_address(op, ctx);

        #[cfg(test)]
        eprintln!(
            "[VLDA] addr=0x{:04X} sources={:?} dest={:?}",
            addr, op.sources, op.dest
        );

        // Read 256 bits (32 bytes) from memory
        let vec_data = Self::read_vector_from_memory(tile, addr);

        // Write to destination vector register
        if let Some(Operand::VectorReg(r)) = &op.dest {
            ctx.vector.write(*r, vec_data);
        }

        // Apply post-modify (vector width = 32 bytes)
        Self::apply_post_modify(op, ctx, post_modify);
    }

    /// Execute vector load B (VLDB).
    ///
    /// Loads 256 bits (32 bytes) from memory into a vector register using the
    /// B-channel pointer (typically p4-p7). Functionally identical to VLDA but
    /// uses a separate address path for parallel memory access.
    fn execute_vector_load_b(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        tile: &Tile,
        post_modify: &PostModify,
    ) {
        let addr = Self::get_address(op, ctx);

        #[cfg(test)]
        eprintln!(
            "[VLDB] addr=0x{:04X} sources={:?} dest={:?}",
            addr, op.sources, op.dest
        );

        // Read 256 bits (32 bytes) from memory
        let vec_data = Self::read_vector_from_memory(tile, addr);

        // Write to destination vector register
        if let Some(Operand::VectorReg(r)) = &op.dest {
            ctx.vector.write(*r, vec_data);
        }

        // Apply post-modify (vector width = 32 bytes)
        Self::apply_post_modify(op, ctx, post_modify);
    }

    /// Execute vector load with unpack (VLDB_UNPACK).
    ///
    /// Loads narrow-type data from memory and unpacks (expands) to wider elements.
    /// For example, loading 8-bit integers and expanding to 16-bit or 32-bit.
    /// This is useful for computation on narrow data types that require wider
    /// intermediate precision.
    fn execute_vector_load_unpack(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        tile: &Tile,
        from_type: ElementType,
        to_type: ElementType,
        post_modify: &PostModify,
    ) {
        let addr = Self::get_address(op, ctx);

        #[cfg(test)]
        eprintln!(
            "[VLDB_UNPACK] addr=0x{:04X} from={:?} to={:?} sources={:?} dest={:?}",
            addr, from_type, to_type, op.sources, op.dest
        );

        // Calculate how many bytes to load based on source element type
        // and the fact that we produce a full 256-bit vector
        let to_bits = to_type.bits() as u32;
        let from_bits = from_type.bits() as u32;

        // Number of destination elements in a 256-bit vector
        let dest_lanes = 256 / to_bits;
        // Number of source bytes needed
        let src_bytes = (dest_lanes * from_bits / 8) as usize;

        let mem = tile.data_memory();
        let addr_usize = addr as usize;

        // Read the packed source data
        let mut result = [0u32; 8];

        if addr_usize + src_bytes <= mem.len() {
            match (from_type, to_type) {
                // 8-bit to 16-bit: read 16 bytes, produce 16 x 16-bit values (packed into 8 x u32)
                // Each u32 holds two 16-bit values: [hi_16 | lo_16]
                (ElementType::Int8, ElementType::Int16) => {
                    for i in 0..8 {
                        // Sign-extend each byte to 16 bits, pack two into one u32
                        let lo_byte = mem[addr_usize + i * 2] as i8 as i16 as u16;
                        let hi_byte = mem[addr_usize + i * 2 + 1] as i8 as i16 as u16;
                        result[i] = ((hi_byte as u32) << 16) | (lo_byte as u32);
                    }
                }
                (ElementType::UInt8, ElementType::Int16) | (ElementType::UInt8, ElementType::UInt16) => {
                    for i in 0..8 {
                        // Zero-extend each byte to 16 bits, pack two into one u32
                        let lo_byte = mem[addr_usize + i * 2] as u32;
                        let hi_byte = mem[addr_usize + i * 2 + 1] as u32;
                        result[i] = (hi_byte << 16) | lo_byte;
                    }
                }
                // 8-bit to 32-bit: read 8 bytes, produce 8 x 32-bit values
                (ElementType::Int8, ElementType::Int32) => {
                    for i in 0..8 {
                        result[i] = mem[addr_usize + i] as i8 as i32 as u32;
                    }
                }
                (ElementType::UInt8, ElementType::UInt32) => {
                    for i in 0..8 {
                        result[i] = mem[addr_usize + i] as u32;
                    }
                }
                // 16-bit to 32-bit: read 16 bytes, produce 8 x 32-bit values
                (ElementType::Int16, ElementType::Int32) => {
                    for i in 0..8 {
                        let offset = addr_usize + i * 2;
                        let val = i16::from_le_bytes([mem[offset], mem[offset + 1]]);
                        result[i] = val as i32 as u32;
                    }
                }
                (ElementType::UInt16, ElementType::UInt32) => {
                    for i in 0..8 {
                        let offset = addr_usize + i * 2;
                        let val = u16::from_le_bytes([mem[offset], mem[offset + 1]]);
                        result[i] = val as u32;
                    }
                }
                // BFloat16 to Float32: read 16 bytes, produce 8 x 32-bit floats
                (ElementType::BFloat16, ElementType::Float32) => {
                    for i in 0..8 {
                        let offset = addr_usize + i * 2;
                        let bf16_bits = u16::from_le_bytes([mem[offset], mem[offset + 1]]);
                        // BF16 to F32: shift left by 16 bits (add 16 zero mantissa bits)
                        result[i] = (bf16_bits as u32) << 16;
                    }
                }
                // Default: just do a straight load with no expansion
                _ => {
                    let vec_data = Self::read_vector_from_memory(tile, addr);
                    result = vec_data;
                }
            }
        }

        // Write to destination vector register
        if let Some(Operand::VectorReg(r)) = &op.dest {
            ctx.vector.write(*r, result);
        }

        // Apply post-modify based on source bytes read
        // For unpack, we advance by the number of bytes actually read, not 32
        match post_modify {
            PostModify::None => {}
            PostModify::Immediate(imm) => {
                // Use the immediate value directly if specified
                if let Some(ptr_reg) = Self::get_pointer_reg(op) {
                    ctx.pointer.add(ptr_reg, *imm as i32);
                }
            }
            PostModify::Register(m) => {
                // Use modifier register
                if let Some(ptr_reg) = Self::get_pointer_reg(op) {
                    let modifier = ctx.modifier.read_signed(*m);
                    ctx.pointer.add(ptr_reg, modifier);
                }
            }
        }
    }

    /// Execute vector store (VST).
    ///
    /// Stores 256 bits (32 bytes) from a vector register to memory.
    fn execute_vector_store(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        tile: &mut Tile,
        post_modify: &PostModify,
    ) {
        let addr = Self::get_address(op, ctx);

        #[cfg(test)]
        eprintln!(
            "[VST] addr=0x{:04X} sources={:?} dest={:?}",
            addr, op.sources, op.dest
        );

        // Get vector register from source or dest
        // For VST, the vector data comes from a source operand (second source)
        // or sometimes the dest field is used for the value
        let vec_reg = op
            .sources
            .get(1)
            .or(op.dest.as_ref())
            .and_then(|operand| match operand {
                Operand::VectorReg(r) => Some(*r),
                _ => None,
            });

        if let Some(r) = vec_reg {
            let vec_data = ctx.vector.read(r);
            Self::write_vector_to_memory(tile, addr, vec_data);
        }

        // Apply post-modify (vector width = 32 bytes)
        Self::apply_post_modify(op, ctx, post_modify);
    }

    /// Get the pointer register from a slot op's source operands.
    fn get_pointer_reg(op: &SlotOp) -> Option<u8> {
        op.sources.first().and_then(|src| match src {
            Operand::Memory { base, .. } => Some(*base),
            Operand::PointerReg(r) => Some(*r),
            _ => None,
        })
    }

    /// Get address from memory operand or pointer register.
    ///
    /// For indexed addressing (ptr + offset), the offset is a word index
    /// that gets scaled by the element size (4 bytes for Word width).
    fn get_address(op: &SlotOp, ctx: &ExecutionContext) -> u32 {
        // First check for Memory operand (encapsulates ptr+offset, already scaled)
        if let Some(Operand::Memory { base, offset }) = op.sources.first() {
            let base_addr = ctx.pointer.read(*base);
            return base_addr.wrapping_add(*offset as i32 as u32);
        }

        // Handle indexed addressing: sources[0]=ptr, sources[1]=offset
        // Offset is a word index, scaled by 4 bytes
        let base_addr = op.sources.first().map_or(0, |src| match src {
            Operand::PointerReg(r) => ctx.pointer.read(*r),
            Operand::ScalarReg(r) => ctx.scalar.read(*r),
            Operand::Immediate(v) => *v as u32,
            _ => 0,
        });

        let offset = op.sources.get(1).map_or(0, |src| match src {
            Operand::Immediate(v) => (*v as i32 * 4) as u32, // Scale by word size
            Operand::ScalarReg(r) => ctx.scalar.read(*r).wrapping_mul(4),
            _ => 0,
        });

        base_addr.wrapping_add(offset)
    }

    /// Get store address.
    ///
    /// Handles two operand layouts:
    /// 1. Decoded kernel: sources[0]=value, sources[1]=ptr, sources[2]=offset
    /// 2. Test/legacy: sources[0]=ptr (no offset)
    ///
    /// Offset is a word index, scaled by 4 bytes.
    fn get_store_address(op: &SlotOp, ctx: &ExecutionContext) -> u32 {
        // First check for Memory operand anywhere (encapsulates ptr+offset)
        for src in &op.sources {
            if let Operand::Memory { base, offset } = src {
                let base_addr = ctx.pointer.read(*base);
                return base_addr.wrapping_add(*offset as i32 as u32);
            }
        }

        // Check if sources[0] is a pointer (test/legacy layout)
        if let Some(Operand::PointerReg(r)) = op.sources.first() {
            return ctx.pointer.read(*r);
        }

        // Otherwise: kernel layout - sources[1] is pointer, sources[2] is offset
        let ptr_val = op.sources.get(1).map_or(0, |src| match src {
            Operand::PointerReg(r) => ctx.pointer.read(*r),
            Operand::ScalarReg(r) => ctx.scalar.read(*r),
            Operand::Immediate(v) => *v as u32,
            _ => 0,
        });

        let offset = op.sources.get(2).map_or(0, |src| match src {
            Operand::Immediate(v) => (*v as i32 * 4) as u32, // Scale by word size
            Operand::ScalarReg(r) => ctx.scalar.read(*r).wrapping_mul(4),
            _ => 0,
        });

        ptr_val.wrapping_add(offset)
    }

    /// Get value to store.
    ///
    /// Handles two operand layouts:
    /// 1. Decoded kernel: sources[0]=value, sources[1]=ptr, sources[2]=offset
    /// 2. Test/legacy: sources[0]=ptr, dest=value
    ///
    /// Uses `ctx.scalar_read()` for VLIW-safe reads that respect the
    /// bundle snapshot when inside a VLIW bundle.
    fn get_store_value(op: &SlotOp, ctx: &ExecutionContext, width: MemWidth) -> u64 {
        // If sources[0] is a pointer/memory operand, value is in dest (test layout)
        // Otherwise, sources[0] is the value (kernel layout)
        let operand = op.sources.first().and_then(|first| {
            match first {
                Operand::PointerReg(_) | Operand::Memory { .. } => op.dest.as_ref(),
                _ => Some(first),
            }
        }).or(op.dest.as_ref());

        operand.map_or(0, |src| match src {
            Operand::ScalarReg(r) => ctx.scalar_read(*r) as u64,
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
    ///
    /// Addresses are masked to LOCAL_MEMORY_MASK to handle linker-assigned
    /// addresses (e.g., 0x70440 -> 0x0440).
    fn read_memory(tile: &Tile, addr: u32, width: MemWidth) -> u64 {
        // Mask address to local memory range (see module docs for AGU addressing)
        let addr = (addr & LOCAL_MEMORY_MASK) as usize;
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
    ///
    /// Addresses are masked to LOCAL_MEMORY_MASK to handle linker-assigned
    /// addresses (e.g., 0x70440 -> 0x0440).
    fn write_memory(tile: &mut Tile, addr: u32, value: u64, width: MemWidth) {
        // Mask address to local memory range (see module docs for AGU addressing)
        let addr = (addr & LOCAL_MEMORY_MASK) as usize;
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
    ///
    /// Addresses are masked to LOCAL_MEMORY_MASK.
    pub fn read_vector_from_memory(tile: &Tile, addr: u32) -> [u32; 8] {
        // Mask address to local memory range
        let addr = (addr & LOCAL_MEMORY_MASK) as usize;
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
    ///
    /// Addresses are masked to LOCAL_MEMORY_MASK.
    pub fn write_vector_to_memory(tile: &mut Tile, addr: u32, value: [u32; 8]) {
        // Mask address to local memory range
        let addr = (addr & LOCAL_MEMORY_MASK) as usize;
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

    #[test]
    fn test_vector256_load() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Write test vector data to memory (256 bits = 8 x 32-bit words)
        let test_data = [
            0x11111111u32,
            0x22222222,
            0x33333333,
            0x44444444,
            0x55555555,
            0x66666666,
            0x77777777,
            0x88888888,
        ];
        MemoryUnit::write_vector_to_memory(&mut tile, 0x400, test_data);

        // Set pointer
        ctx.pointer.write(0, 0x400);

        // Load vector: v0 = [p0]
        let op = SlotOp::new(
            SlotIndex::Load,
            Operation::Load {
                width: MemWidth::Vector256,
                post_modify: PostModify::None,
            },
        )
        .with_dest(Operand::VectorReg(0))
        .with_source(Operand::PointerReg(0));

        MemoryUnit::execute(&op, &mut ctx, &mut tile);

        // Verify all 8 lanes were loaded correctly
        let loaded = ctx.vector.read(0);
        assert_eq!(loaded, test_data);
    }

    #[test]
    fn test_vector256_store() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Set up vector register with test data
        let test_data = [
            0xAAAA_BBBBu32,
            0xCCCC_DDDD,
            0xEEEE_FFFF,
            0x1111_2222,
            0x3333_4444,
            0x5555_6666,
            0x7777_8888,
            0x9999_0000,
        ];
        ctx.vector.write(1, test_data);

        // Set pointer
        ctx.pointer.write(0, 0x500);

        // Store vector: [p0] = v1
        let op = SlotOp::new(
            SlotIndex::Store,
            Operation::Store {
                width: MemWidth::Vector256,
                post_modify: PostModify::None,
            },
        )
        .with_dest(Operand::VectorReg(1))
        .with_source(Operand::PointerReg(0));

        MemoryUnit::execute(&op, &mut ctx, &mut tile);

        // Verify all 8 lanes were stored correctly
        let stored = MemoryUnit::read_vector_from_memory(&tile, 0x500);
        assert_eq!(stored, test_data);
    }

    #[test]
    fn test_vector256_load_with_post_modify() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        let test_data = [1, 2, 3, 4, 5, 6, 7, 8];
        MemoryUnit::write_vector_to_memory(&mut tile, 0x600, test_data);
        ctx.pointer.write(0, 0x600);

        // Load with post-modify: v0 = [p0], p0 += 32
        let op = SlotOp::new(
            SlotIndex::Load,
            Operation::Load {
                width: MemWidth::Vector256,
                post_modify: PostModify::Immediate(32), // 256 bits = 32 bytes
            },
        )
        .with_dest(Operand::VectorReg(0))
        .with_source(Operand::PointerReg(0));

        MemoryUnit::execute(&op, &mut ctx, &mut tile);

        assert_eq!(ctx.vector.read(0), test_data);
        assert_eq!(ctx.pointer.read(0), 0x620); // 0x600 + 32
    }

    #[test]
    fn test_vector_load_a() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Write test vector data
        let test_data = [0xAABBCCDD, 0x11223344, 0x55667788, 0x99AABBCC,
                         0xDDEEFF00, 0x12345678, 0x9ABCDEF0, 0xFEDCBA98];
        MemoryUnit::write_vector_to_memory(&mut tile, 0x700, test_data);
        ctx.pointer.write(0, 0x700);

        // VLDA: v2 = [p0]
        let op = SlotOp::new(
            SlotIndex::Load,
            Operation::VectorLoadA {
                post_modify: PostModify::None,
            },
        )
        .with_dest(Operand::VectorReg(2))
        .with_source(Operand::PointerReg(0));

        assert!(MemoryUnit::execute(&op, &mut ctx, &mut tile));
        assert_eq!(ctx.vector.read(2), test_data);
        assert_eq!(ctx.pointer.read(0), 0x700); // No post-modify
    }

    #[test]
    fn test_vector_load_a_with_post_modify() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        let test_data = [1, 2, 3, 4, 5, 6, 7, 8];
        MemoryUnit::write_vector_to_memory(&mut tile, 0x800, test_data);
        ctx.pointer.write(1, 0x800);

        // VLDA: v0 = [p1], p1 += 32
        let op = SlotOp::new(
            SlotIndex::Load,
            Operation::VectorLoadA {
                post_modify: PostModify::Immediate(32),
            },
        )
        .with_dest(Operand::VectorReg(0))
        .with_source(Operand::PointerReg(1));

        MemoryUnit::execute(&op, &mut ctx, &mut tile);
        assert_eq!(ctx.vector.read(0), test_data);
        assert_eq!(ctx.pointer.read(1), 0x820); // 0x800 + 32
    }

    #[test]
    fn test_vector_load_b() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        let test_data = [0xDEADBEEF, 0xCAFEBABE, 0x12345678, 0x9ABCDEF0,
                         0x0F0F0F0F, 0xF0F0F0F0, 0xAAAA5555, 0x5555AAAA];
        MemoryUnit::write_vector_to_memory(&mut tile, 0x900, test_data);
        ctx.pointer.write(4, 0x900); // B-channel typically uses p4-p7

        // VLDB: v3 = [p4]
        let op = SlotOp::new(
            SlotIndex::Load,
            Operation::VectorLoadB {
                post_modify: PostModify::None,
            },
        )
        .with_dest(Operand::VectorReg(3))
        .with_source(Operand::PointerReg(4));

        assert!(MemoryUnit::execute(&op, &mut ctx, &mut tile));
        assert_eq!(ctx.vector.read(3), test_data);
    }

    #[test]
    fn test_vector_load_b_with_modifier_register() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        let test_data = [10, 20, 30, 40, 50, 60, 70, 80];
        MemoryUnit::write_vector_to_memory(&mut tile, 0xA00, test_data);
        ctx.pointer.write(5, 0xA00);
        ctx.modifier.write(2, 64); // m2 = 64

        // VLDB: v1 = [p5], p5 += m2
        let op = SlotOp::new(
            SlotIndex::Load,
            Operation::VectorLoadB {
                post_modify: PostModify::Register(2),
            },
        )
        .with_dest(Operand::VectorReg(1))
        .with_source(Operand::PointerReg(5));

        MemoryUnit::execute(&op, &mut ctx, &mut tile);
        assert_eq!(ctx.vector.read(1), test_data);
        assert_eq!(ctx.pointer.read(5), 0xA40); // 0xA00 + 64
    }

    #[test]
    fn test_vector_store() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        let test_data = [0x11111111, 0x22222222, 0x33333333, 0x44444444,
                         0x55555555, 0x66666666, 0x77777777, 0x88888888];
        ctx.vector.write(4, test_data);
        ctx.pointer.write(2, 0xB00);

        // VST: [p2] = v4
        let op = SlotOp::new(
            SlotIndex::Store,
            Operation::VectorStore {
                post_modify: PostModify::None,
            },
        )
        .with_dest(Operand::VectorReg(4))
        .with_source(Operand::PointerReg(2));

        assert!(MemoryUnit::execute(&op, &mut ctx, &mut tile));

        let stored = MemoryUnit::read_vector_from_memory(&tile, 0xB00);
        assert_eq!(stored, test_data);
    }

    #[test]
    fn test_vector_store_with_post_modify() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        let test_data = [100, 200, 300, 400, 500, 600, 700, 800];
        ctx.vector.write(5, test_data);
        ctx.pointer.write(3, 0xC00);

        // VST: [p3] = v5, p3 += 32
        let op = SlotOp::new(
            SlotIndex::Store,
            Operation::VectorStore {
                post_modify: PostModify::Immediate(32),
            },
        )
        .with_dest(Operand::VectorReg(5))
        .with_source(Operand::PointerReg(3));

        MemoryUnit::execute(&op, &mut ctx, &mut tile);

        let stored = MemoryUnit::read_vector_from_memory(&tile, 0xC00);
        assert_eq!(stored, test_data);
        assert_eq!(ctx.pointer.read(3), 0xC20); // 0xC00 + 32
    }

    #[test]
    fn test_vector_load_unpack_int8_to_int32() {
        use crate::interpreter::bundle::ElementType;

        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Write 8 bytes of int8 data
        let mem = tile.data_memory_mut();
        for i in 0..8 {
            mem[0xD00 + i] = (i + 1) as u8; // 1, 2, 3, 4, 5, 6, 7, 8
        }
        ctx.pointer.write(0, 0xD00);

        // VLDB_UNPACK: expand int8 to int32
        let op = SlotOp::new(
            SlotIndex::Load,
            Operation::VectorLoadUnpack {
                from_type: ElementType::UInt8,
                to_type: ElementType::UInt32,
                post_modify: PostModify::None,
            },
        )
        .with_dest(Operand::VectorReg(6))
        .with_source(Operand::PointerReg(0));

        assert!(MemoryUnit::execute(&op, &mut ctx, &mut tile));

        let result = ctx.vector.read(6);
        // Each byte should be zero-extended to 32 bits
        assert_eq!(result[0], 1);
        assert_eq!(result[1], 2);
        assert_eq!(result[2], 3);
        assert_eq!(result[3], 4);
        assert_eq!(result[4], 5);
        assert_eq!(result[5], 6);
        assert_eq!(result[6], 7);
        assert_eq!(result[7], 8);
    }

    #[test]
    fn test_vector_load_unpack_int8_to_int32_signed() {
        use crate::interpreter::bundle::ElementType;

        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Write 8 bytes with negative values
        let mem = tile.data_memory_mut();
        mem[0xE00] = 0xFF; // -1
        mem[0xE01] = 0xFE; // -2
        mem[0xE02] = 0x7F; // 127
        mem[0xE03] = 0x80; // -128
        mem[0xE04] = 0x01; // 1
        mem[0xE05] = 0x00; // 0
        mem[0xE06] = 0xF0; // -16
        mem[0xE07] = 0x10; // 16
        ctx.pointer.write(1, 0xE00);

        // VLDB_UNPACK: expand signed int8 to int32
        let op = SlotOp::new(
            SlotIndex::Load,
            Operation::VectorLoadUnpack {
                from_type: ElementType::Int8,
                to_type: ElementType::Int32,
                post_modify: PostModify::None,
            },
        )
        .with_dest(Operand::VectorReg(7))
        .with_source(Operand::PointerReg(1));

        MemoryUnit::execute(&op, &mut ctx, &mut tile);

        let result = ctx.vector.read(7);
        assert_eq!(result[0] as i32, -1);
        assert_eq!(result[1] as i32, -2);
        assert_eq!(result[2] as i32, 127);
        assert_eq!(result[3] as i32, -128);
        assert_eq!(result[4] as i32, 1);
        assert_eq!(result[5] as i32, 0);
        assert_eq!(result[6] as i32, -16);
        assert_eq!(result[7] as i32, 16);
    }

    #[test]
    fn test_vector_load_unpack_int16_to_int32() {
        use crate::interpreter::bundle::ElementType;

        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Write 16 bytes of int16 data (8 values)
        let mem = tile.data_memory_mut();
        let values: [i16; 8] = [100, -100, 32767, -32768, 0, 1, -1, 12345];
        for i in 0..8 {
            let bytes = values[i].to_le_bytes();
            mem[0xF00 + i * 2] = bytes[0];
            mem[0xF00 + i * 2 + 1] = bytes[1];
        }
        ctx.pointer.write(2, 0xF00);

        // VLDB_UNPACK: expand int16 to int32
        let op = SlotOp::new(
            SlotIndex::Load,
            Operation::VectorLoadUnpack {
                from_type: ElementType::Int16,
                to_type: ElementType::Int32,
                post_modify: PostModify::Immediate(16), // Read 16 bytes
            },
        )
        .with_dest(Operand::VectorReg(0))
        .with_source(Operand::PointerReg(2));

        MemoryUnit::execute(&op, &mut ctx, &mut tile);

        let result = ctx.vector.read(0);
        assert_eq!(result[0] as i32, 100);
        assert_eq!(result[1] as i32, -100);
        assert_eq!(result[2] as i32, 32767);
        assert_eq!(result[3] as i32, -32768);
        assert_eq!(result[4] as i32, 0);
        assert_eq!(result[5] as i32, 1);
        assert_eq!(result[6] as i32, -1);
        assert_eq!(result[7] as i32, 12345);
        assert_eq!(ctx.pointer.read(2), 0xF10); // 0xF00 + 16
    }
}
