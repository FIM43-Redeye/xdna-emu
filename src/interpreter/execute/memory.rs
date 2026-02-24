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
//! - 0x00000-0x0FFFF: Local tile memory (quadrant 0)
//! - 0x10000-0x1FFFF: West neighbor (col-1, same row)
//! - 0x20000-0x2FFFF: North neighbor (same col, row+1)
//! - 0x30000-0x3FFFF: East neighbor (col+1, same row)
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
use crate::interpreter::timing::{LATENCY_MEMORY, CROSS_TILE_LATENCY};

/// Address mask for local tile memory (64KB).
/// Addresses from the AGU span 0x0000-0x3FFFF (256KB across 4 neighbors),
/// but local memory access only uses the lower 16 bits.
const LOCAL_MEMORY_MASK: u32 = 0xFFFF;

/// Maximum valid data memory address for the 18-bit quadrant address space.
/// Addresses above this come from the linker's system address map (e.g.,
/// 0x70000 for a compute tile) and should be masked to a 16-bit local offset.
const DATA_MEMORY_ADDRESS_LIMIT: u32 = 0x3FFFF;

/// Extract the quadrant (0-3) and local offset from a data memory address.
///
/// AIE2 cores use an 18-bit address space for data memory:
///   - bits 17:16 select the quadrant (0=Local, 1=West, 2=North, 3=East)
///   - bits 15:0  are the 64KB offset within the quadrant
///
/// Quadrant 0 is treated as local memory. Linker-assigned system addresses
/// (e.g. 0x70000 for East/local) are masked to 16-bit offsets, also landing
/// in quadrant 0.
///
/// However, addresses loaded from the ELF/linker may include the tile's
/// system base address (e.g., 0x70000 for tile at col 0, row 2). These
/// addresses exceed the 18-bit window and must be masked to a 16-bit local
/// offset (quadrant 0).
#[inline]
fn decode_data_address(addr: u32) -> (u32, usize) {
    if addr > DATA_MEMORY_ADDRESS_LIMIT {
        // Linker/system address -- strip to local 16-bit offset
        (0, (addr & LOCAL_MEMORY_MASK) as usize)
    } else {
        // Valid 18-bit data memory address -- use quadrant routing
        let quadrant = (addr >> 16) & 0x3;
        let offset = (addr & LOCAL_MEMORY_MASK) as usize;
        (quadrant, offset)
    }
}

/// Compute the total load latency for a data memory address.
///
/// Local memory (quadrant 0) uses the base LATENCY_MEMORY (7 cycles).
/// Cross-tile accesses (quadrants 1-3) add CROSS_TILE_LATENCY (4 cycles)
/// per hop to model the routing delay through the stream switch.
#[inline]
fn load_latency_for_address(addr: u32) -> u64 {
    let (quadrant, _) = decode_data_address(addr);
    let base = LATENCY_MEMORY as u64;
    if quadrant == 0 {
        base
    } else {
        base + CROSS_TILE_LATENCY as u64
    }
}

/// Read-only snapshots of neighbor tile data memory for cross-tile access.
///
/// Built lazily before each core step. Neighbor memory is only cloned on
/// the first cross-tile access, so cores that only touch local memory
/// (quadrant 0) pay zero allocation cost.
///
/// Cross-tile writes are buffered and applied after the core step completes,
/// matching hardware behavior where cross-tile writes have higher latency
/// and become visible on the next cycle.
///
/// # Quadrant Mapping
///
/// | Quadrant | Direction | Neighbor offset        |
/// |----------|-----------|------------------------|
/// | 0        | Local     | Own tile memory         |
/// | 1        | West      | (col-1, row)           |
/// | 2        | North     | (col, row+1)           |
/// | 3        | East      | (col+1, row)           |
///
/// Note: the hardware's AGU quadrant numbering per aie-rt/mlir-aie is
/// 0=South, 1=West, 2=North, 3=East(local). The emulator uses quadrant 0
/// as local because linker-assigned addresses (0x70000+) are masked to
/// 16-bit offsets landing in quadrant 0 (see [`decode_data_address`]).
pub struct NeighborMemory {
    /// Source tile coordinates (needed for resolving neighbors).
    col: usize,
    row: usize,

    /// Neighbor data memory snapshots indexed by quadrant-1 (0=West, 1=North, 2=East).
    /// None until first access (lazy clone). Inner None if neighbor doesn't exist.
    snapshots: [Option<Option<Vec<u8>>>; 3],

    /// Buffered cross-tile writes: (quadrant, offset_within_tile, data).
    /// Applied after core step completes.
    pub pending_writes: Vec<(u8, usize, Vec<u8>)>,
}

impl NeighborMemory {
    /// Create a new NeighborMemory for the core at (col, row).
    ///
    /// Does NOT clone any neighbor memory yet -- snapshots are lazy.
    pub fn new(col: usize, row: usize) -> Self {
        Self {
            col,
            row,
            snapshots: [None, None, None],
            pending_writes: Vec::new(),
        }
    }

    /// Resolve a quadrant (1-3) to the neighbor tile coordinates.
    ///
    /// Returns None if the neighbor is outside the array bounds.
    fn neighbor_coords(&self, quadrant: u8) -> Option<(usize, usize)> {
        match quadrant {
            1 => {
                // West: col-1, same row
                if self.col > 0 { Some((self.col - 1, self.row)) } else { None }
            }
            2 => {
                // North: same col, row+1
                Some((self.col, self.row + 1))
            }
            3 => {
                // East: col+1, same row
                Some((self.col + 1, self.row))
            }
            _ => None,
        }
    }

    /// Lazily snapshot a neighbor tile's data memory.
    ///
    /// Called internally on first cross-tile access for a given quadrant.
    /// The `device` reference is only needed for this initial clone.
    pub fn ensure_snapshot(&mut self, quadrant: u8, device: &crate::device::DeviceState) {
        let idx = (quadrant - 1) as usize;
        if idx >= 3 || self.snapshots[idx].is_some() {
            return; // Already loaded or invalid quadrant
        }

        let snapshot = self.neighbor_coords(quadrant)
            .and_then(|(c, r)| device.tile(c, r))
            .map(|tile| tile.data_memory().to_vec());

        self.snapshots[idx] = Some(snapshot);
    }

    /// Get a reference to a neighbor's data memory snapshot.
    ///
    /// Returns None if the neighbor doesn't exist or hasn't been snapshotted.
    pub fn get_memory(&self, quadrant: u8) -> Option<&[u8]> {
        let idx = (quadrant - 1) as usize;
        if idx >= 3 {
            return None;
        }
        self.snapshots[idx]
            .as_ref()
            .and_then(|opt| opt.as_deref())
    }

    /// Buffer a cross-tile write for deferred application.
    pub fn buffer_write(&mut self, quadrant: u8, offset: usize, data: &[u8]) {
        self.pending_writes.push((quadrant, offset, data.to_vec()));
    }

    /// Apply all buffered cross-tile writes to the device.
    ///
    /// Resolves each quadrant to the target tile and writes the data.
    /// Called after the core step completes.
    pub fn apply_writes(self, device: &mut crate::device::DeviceState) {
        let col = self.col;
        let row = self.row;
        for (quadrant, offset, data) in self.pending_writes {
            // Resolve quadrant to neighbor coordinates inline (self is consumed)
            let coords = match quadrant {
                1 => if col > 0 { Some((col - 1, row)) } else { None },
                2 => Some((col, row + 1)),
                3 => Some((col + 1, row)),
                _ => None,
            };
            if let Some((c, r)) = coords {
                if let Some(tile) = device.tile_mut(c, r) {
                    let mem = tile.data_memory_mut();
                    if offset + data.len() <= mem.len() {
                        mem[offset..offset + data.len()].copy_from_slice(&data);
                    }
                }
            }
        }
    }

    /// Check if there are any pending cross-tile writes.
    pub fn has_pending_writes(&self) -> bool {
        !self.pending_writes.is_empty()
    }
}

/// Memory unit for load/store operations.
pub struct MemoryUnit;

impl MemoryUnit {
    /// Execute a memory operation.
    ///
    /// The `neighbors` parameter enables cross-tile memory access (quadrants 1-3).
    /// Pass `None` for local-only access (unit tests, simple mode).
    ///
    /// Returns `true` if the operation was handled, `false` if not a memory op.
    pub fn execute(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        tile: &mut Tile,
        neighbors: Option<&mut NeighborMemory>,
    ) -> bool {
        // PostModify comes from op.post_modify (populated directly from the
        // AG field during decode) rather than from the Operation variant.
        let pm = &op.post_modify;

        match &op.op {
            Operation::Load { width, .. } => {
                Self::execute_load(op, ctx, tile, *width, pm, neighbors);
                true
            }

            Operation::Store { width, .. } => {
                Self::execute_store(op, ctx, tile, *width, pm, neighbors);
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

            Operation::VectorLoadA { .. } => {
                Self::execute_vector_load_a(op, ctx, tile, pm, neighbors);
                true
            }

            Operation::VectorLoadB { .. } => {
                Self::execute_vector_load_b(op, ctx, tile, pm, neighbors);
                true
            }

            Operation::VectorLoadUnpack {
                from_type,
                to_type,
                ..
            } => {
                Self::execute_vector_load_unpack(op, ctx, tile, *from_type, *to_type, pm);
                true
            }

            Operation::VectorStore { .. } => {
                Self::execute_vector_store(op, ctx, tile, pm, neighbors);
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
        let ptr_value = ctx.pointer_read(ptr_reg);

        // Get offset from source operand
        let offset = match op.sources.first() {
            Some(Operand::Immediate(imm)) => *imm as i32,
            Some(Operand::ScalarReg(r)) => ctx.scalar_read(*r) as i32,
            Some(Operand::ModifierReg(r)) => ctx.modifier_read(*r) as i32,
            _ => 0,
        };

        // Add offset to pointer (wrapping)
        let new_value = (ptr_value as i32).wrapping_add(offset) as u32;

        ctx.pointer.write(ptr_reg, new_value);
    }

    /// Execute pointer move: ptr = value.
    /// mova, movb instructions set pointer registers.
    fn execute_pointer_mov(op: &SlotOp, ctx: &mut ExecutionContext) {
        // Get destination pointer register
        let ptr_reg = match &op.dest {
            Some(Operand::PointerReg(r)) => *r,
            _ => return,
        };

        // Get value from source operand
        let value = match op.sources.first() {
            Some(Operand::Immediate(imm)) => *imm as u32,
            Some(Operand::ScalarReg(r)) => ctx.scalar_read(*r),
            Some(Operand::PointerReg(r)) => ctx.pointer_read(*r),
            _ => 0,
        };

        ctx.pointer.write(ptr_reg, value);
    }

    /// Execute a load operation.
    fn execute_load(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        tile: &Tile,
        width: MemWidth,
        post_modify: &PostModify,
        neighbors: Option<&mut NeighborMemory>,
    ) {
        // Get address from source operand
        let addr = Self::get_address(op, ctx);
        let latency = load_latency_for_address(addr);

        // Track bank access for conflict detection (local memory only)
        let (quadrant, local_offset) = decode_data_address(addr);
        if quadrant == 0 {
            ctx.record_core_bank_access(local_offset as u32, width.bytes() as usize, tile.num_banks());
        }

        log::trace!("[LOAD] addr=0x{:X} width={:?} dest={:?} srcs={:?} latency={}",
            addr, width, op.dest, op.sources, latency);

        // Handle full vector loads specially
        if width == MemWidth::Vector256 {
            let vec_data = Self::read_vector_from_memory(tile, addr, neighbors.map(|n| &*n));
            if let Some(dest) = &op.dest {
                ctx.queue_vector_load(dest.clone(), vec_data, latency);
            }
        } else {
            // Scalar/partial loads go through write_dest_with_latency (which queues)
            let value = Self::read_memory(tile, addr, width, neighbors.map(|n| &*n));
            if log::log_enabled!(log::Level::Trace) {
                let masked = addr & 0xFFFF;
                if masked >= 0x0400 && masked < 0x0420 {
                    let elem = (masked - 0x0400) / 4;
                    log::trace!(
                        "[WATCH-LD] pc=0x{:03X} cycle={} elem={} addr=0x{:05X} value={} dest={:?}",
                        ctx.pc(), ctx.cycles, elem, masked, value as i32, op.dest
                    );
                }
            }
            Self::write_dest_with_latency(op, ctx, value, width, latency);
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
        neighbors: Option<&mut NeighborMemory>,
    ) {
        // Get address using store-specific layout: sources[1]=ptr, sources[2]=offset
        let addr = Self::get_store_address(op, ctx);

        // Track bank access for conflict detection (local memory only)
        let (quadrant, local_offset) = decode_data_address(addr);
        if quadrant == 0 {
            ctx.record_core_bank_access(local_offset as u32, width.bytes() as usize, tile.num_banks());
        }

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
                Self::write_vector_to_memory(tile, addr, vec_data, neighbors);
            }
        } else {
            // Scalar/partial stores
            let value = Self::get_store_value(op, ctx, width);
            if log::log_enabled!(log::Level::Trace) {
                let masked = addr & 0xFFFF;
                if masked >= 0x0400 && masked < 0x0420 {
                    let elem = (masked - 0x0400) / 4;
                    log::trace!(
                        "[WATCH-ST] pc=0x{:03X} cycle={} elem={} addr=0x{:05X} value={} srcs={:?}",
                        ctx.pc(), ctx.cycles, elem, masked, value as i32, op.sources
                    );
                }
            }
            Self::write_memory(tile, addr, value, width, neighbors);
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
        neighbors: Option<&mut NeighborMemory>,
    ) {
        let addr = Self::get_address(op, ctx);
        let latency = load_latency_for_address(addr);

        // Track bank access for conflict detection (local memory only)
        let (quadrant, local_offset) = decode_data_address(addr);
        if quadrant == 0 {
            ctx.record_core_bank_access(local_offset as u32, 32, tile.num_banks());
        }

        // Read 256 bits (32 bytes) from memory
        let vec_data = Self::read_vector_from_memory(tile, addr, neighbors.map(|n| &*n));

        // Queue deferred write to destination vector register
        if let Some(dest) = &op.dest {
            ctx.queue_vector_load(dest.clone(), vec_data, latency);
        }

        // Apply post-modify immediately (pointer update is not deferred)
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
        neighbors: Option<&mut NeighborMemory>,
    ) {
        let addr = Self::get_address(op, ctx);
        let latency = load_latency_for_address(addr);

        // Track bank access for conflict detection (local memory only)
        let (quadrant, local_offset) = decode_data_address(addr);
        if quadrant == 0 {
            ctx.record_core_bank_access(local_offset as u32, 32, tile.num_banks());
        }

        // Read 256 bits (32 bytes) from memory
        let vec_data = Self::read_vector_from_memory(tile, addr, neighbors.map(|n| &*n));

        // Queue deferred write to destination vector register
        if let Some(dest) = &op.dest {
            ctx.queue_vector_load(dest.clone(), vec_data, latency);
        }

        // Apply post-modify immediately (pointer update is not deferred)
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
        let latency = load_latency_for_address(addr);

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
                    let vec_data = Self::read_vector_from_memory(tile, addr, None);
                    result = vec_data;
                }
            }
        }

        // Queue deferred write to destination vector register
        if let Some(dest) = &op.dest {
            ctx.queue_vector_load(dest.clone(), result, latency);
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
                // Use modifier register (VLIW-safe read)
                if let Some(ptr_reg) = Self::get_pointer_reg(op) {
                    let modifier = ctx.modifier_read(*m) as i32;
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
        neighbors: Option<&mut NeighborMemory>,
    ) {
        let addr = Self::get_address(op, ctx);

        // Track bank access for conflict detection (local memory only)
        let (quadrant, local_offset) = decode_data_address(addr);
        if quadrant == 0 {
            ctx.record_core_bank_access(local_offset as u32, 32, tile.num_banks());
        }

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
            Self::write_vector_to_memory(tile, addr, vec_data, neighbors);
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
    /// Three operand layouts are handled:
    /// 1. `Memory { base, offset }` -- pre-scaled byte offset from AG decode
    /// 2. `[PointerReg, ModifierReg]` -- indexed register (byte offset in modifier)
    /// 3. `[PointerReg, Immediate/ScalarReg]` -- word-scaled offset
    fn get_address(op: &SlotOp, ctx: &ExecutionContext) -> u32 {
        // First check for Memory operand (encapsulates ptr+offset, already scaled)
        if let Some(Operand::Memory { base, offset }) = op.sources.first() {
            let base_addr = ctx.pointer_read(*base);
            let result = base_addr.wrapping_add(*offset as i32 as u32);
            return result;
        }

        // Handle indexed addressing: sources[0]=ptr, sources[1]=offset
        let base_addr = op.sources.first().map_or(0, |src| match src {
            Operand::PointerReg(r) => ctx.pointer_read(*r),
            Operand::ScalarReg(r) => ctx.scalar_read(*r),
            Operand::Immediate(v) => *v as u32,
            _ => 0,
        });

        let offset = op.sources.get(1).map_or(0, |src| match src {
            Operand::Immediate(v) => (*v as i32 * 4) as u32, // Scale by word size
            Operand::ScalarReg(r) => ctx.scalar_read(*r).wrapping_mul(4),
            // Modifier registers contain byte offsets (set via mov dj0/m0, rN)
            Operand::ModifierReg(r) => ctx.modifier_read(*r),
            _ => 0,
        });

        let result = base_addr.wrapping_add(offset);
        result
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
                let base_addr = ctx.pointer_read(*base);
                return base_addr.wrapping_add(*offset as i32 as u32);
            }
        }

        // Check if sources[0] is a pointer (test/legacy layout)
        if let Some(Operand::PointerReg(r)) = op.sources.first() {
            return ctx.pointer_read(*r);
        }

        // Kernel layout: sources[0]=value, sources[1]=pointer, sources[2]=post-modify
        // The store address is just the pointer value. sources[2] is the
        // post-modify amount (applied after the store), NOT a pre-offset.
        op.sources.get(1).map_or(0, |src| match src {
            Operand::PointerReg(r) => ctx.pointer_read(*r),
            Operand::ScalarReg(r) => ctx.scalar_read(*r),
            Operand::Immediate(v) => *v as u32,
            _ => 0,
        })
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

    /// Read from tile memory with cross-tile quadrant routing.
    ///
    /// Address bits 17:16 select the memory quadrant:
    /// - 0: Local tile memory (fast path)
    /// - 1: West neighbor
    /// - 2: North neighbor
    /// - 3: East neighbor
    ///
    /// Linker-assigned addresses (e.g., 0x70440) are masked to the 16-bit
    /// local offset within the selected quadrant.
    fn read_memory(tile: &Tile, addr: u32, width: MemWidth, neighbors: Option<&NeighborMemory>) -> u64 {
        let (quadrant, offset) = decode_data_address(addr);

        // Select memory source based on quadrant
        let mem: &[u8] = if quadrant == 0 || neighbors.is_none() {
            // Fast path: local tile memory
            tile.data_memory()
        } else if let Some(neighbor_mem) = neighbors.and_then(|n| n.get_memory(quadrant as u8)) {
            neighbor_mem
        } else {
            // Neighbor doesn't exist (edge of array) - return zero
            log::trace!("[LOAD] cross-tile read to non-existent neighbor quadrant {}", quadrant);
            return 0;
        };

        Self::read_from_slice(mem, offset, width)
    }

    /// Read a value from a memory slice at the given offset.
    fn read_from_slice(mem: &[u8], addr: usize, width: MemWidth) -> u64 {
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

    /// Write to tile memory with cross-tile quadrant routing.
    ///
    /// Address bits 17:16 select the memory quadrant. Cross-tile writes
    /// (quadrants 1-3) are buffered in the NeighborMemory for deferred
    /// application after the core step completes.
    fn write_memory(tile: &mut Tile, addr: u32, value: u64, width: MemWidth, neighbors: Option<&mut NeighborMemory>) {
        let (quadrant, offset) = decode_data_address(addr);

        if quadrant != 0 {
            if let Some(nbr) = neighbors {
                // Cross-tile write: serialize value and buffer it
                let data = Self::serialize_value(value, width);
                nbr.buffer_write(quadrant as u8, offset, &data);
                return;
            }
            // No neighbor context -- fall through to local write (legacy behavior)
        }

        // Local tile write (quadrant 0 or no neighbor context)
        let mem = tile.data_memory_mut();
        Self::write_to_slice(mem, offset, value, width);
    }

    /// Serialize a scalar value to bytes for the given width.
    fn serialize_value(value: u64, width: MemWidth) -> Vec<u8> {
        match width {
            MemWidth::Byte => vec![value as u8],
            MemWidth::HalfWord => (value as u16).to_le_bytes().to_vec(),
            MemWidth::Word => (value as u32).to_le_bytes().to_vec(),
            MemWidth::DoubleWord => value.to_le_bytes().to_vec(),
            MemWidth::QuadWord => {
                let mut data = value.to_le_bytes().to_vec();
                data.extend_from_slice(&[0u8; 8]);
                data
            }
            MemWidth::Vector256 => {
                let mut data = value.to_le_bytes().to_vec();
                data.extend_from_slice(&[0u8; 24]);
                data
            }
        }
    }

    /// Write a value to a memory slice at the given offset.
    fn write_to_slice(mem: &mut [u8], addr: usize, value: u64, width: MemWidth) {
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

    /// Queue a loaded scalar value for deferred write to the destination register.
    ///
    /// Load results are deferred by the given latency to model the AIE2
    /// memory pipeline. Cross-tile accesses (quadrants 1-3) incur additional
    /// routing latency on top of the base LATENCY_MEMORY. The compiler relies
    /// on this latency to pipeline multiple loads to the same register.
    fn write_dest_with_latency(op: &SlotOp, ctx: &mut ExecutionContext, value: u64, width: MemWidth, latency: u64) {
        if let Some(dest) = &op.dest {
            match dest {
                Operand::ScalarReg(_)
                | Operand::PointerReg(_)
                | Operand::ModifierReg(_) => {
                    ctx.queue_scalar_load(dest.clone(), value as u32, latency);
                }
                Operand::VectorReg(_) => {
                    if width == MemWidth::Vector256 {
                        let mut vec = [0u32; 8];
                        vec[0] = value as u32;
                        vec[1] = (value >> 32) as u32;
                        ctx.queue_vector_load(dest.clone(), vec, latency);
                    } else {
                        let mut vec = [0u32; 8];
                        vec[0] = value as u32;
                        ctx.queue_vector_load(dest.clone(), vec, latency);
                    }
                }
                _ => {}
            }
        }
    }

    /// Apply post-modify to the base address register.
    fn apply_post_modify(op: &SlotOp, ctx: &mut ExecutionContext, post_modify: &PostModify) {
        // Find the pointer register used as base address.
        // For loads: sources[0] is typically the pointer.
        // For stores: sources[0] is the value, sources[1] is the pointer.
        // Search all sources for the first PointerReg or Memory operand.
        let ptr_reg = op.sources.iter().find_map(|src| match src {
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
                    let modifier = ctx.modifier_read(*m) as i32;
                    ctx.pointer.add(reg, modifier);
                }
            }
        }
    }

    /// Read a vector from memory (256 bits = 32 bytes) with cross-tile routing.
    pub fn read_vector_from_memory(tile: &Tile, addr: u32, neighbors: Option<&NeighborMemory>) -> [u32; 8] {
        let (quadrant, offset) = decode_data_address(addr);

        // Select memory source
        let mem: &[u8] = if quadrant == 0 || neighbors.is_none() {
            tile.data_memory()
        } else if let Some(neighbor_mem) = neighbors.and_then(|n| n.get_memory(quadrant as u8)) {
            neighbor_mem
        } else {
            return [0u32; 8];
        };

        Self::read_vector_from_slice(mem, offset)
    }

    /// Read 8 x u32 from a memory slice at the given byte offset.
    fn read_vector_from_slice(mem: &[u8], addr: usize) -> [u32; 8] {
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

    /// Write a vector to memory (256 bits = 32 bytes) with cross-tile routing.
    pub fn write_vector_to_memory(tile: &mut Tile, addr: u32, value: [u32; 8], neighbors: Option<&mut NeighborMemory>) {
        let (quadrant, offset) = decode_data_address(addr);

        if quadrant != 0 {
            if let Some(nbr) = neighbors {
                // Cross-tile vector write: serialize all 8 words and buffer
                let mut data = Vec::with_capacity(32);
                for word in &value {
                    data.extend_from_slice(&word.to_le_bytes());
                }
                nbr.buffer_write(quadrant as u8, offset, &data);
                return;
            }
        }

        // Local tile write
        let mem = tile.data_memory_mut();
        Self::write_vector_to_slice(mem, offset, value);
    }

    /// Write 8 x u32 to a memory slice at the given byte offset.
    fn write_vector_to_slice(mem: &mut [u8], addr: usize, value: [u32; 8]) {
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

        assert!(MemoryUnit::execute(&op, &mut ctx, &mut tile, None));
        ctx.flush_pending_writes();
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

        MemoryUnit::execute(&op, &mut ctx, &mut tile, None);
        ctx.flush_pending_writes();
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
        .with_post_modify(PostModify::Immediate(4))
        .with_dest(Operand::ScalarReg(0))
        .with_source(Operand::PointerReg(0));

        MemoryUnit::execute(&op, &mut ctx, &mut tile, None);
        ctx.flush_pending_writes();
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
        .with_post_modify(PostModify::Register(0))
        .with_dest(Operand::ScalarReg(0))
        .with_source(Operand::PointerReg(0));

        MemoryUnit::execute(&op, &mut ctx, &mut tile, None);
        ctx.flush_pending_writes();
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

        MemoryUnit::execute(&op, &mut ctx, &mut tile, None);
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

        MemoryUnit::execute(&op, &mut ctx, &mut tile, None);
        ctx.flush_pending_writes();
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

        MemoryUnit::execute(&op, &mut ctx, &mut tile, None);
        ctx.flush_pending_writes();
        assert_eq!(ctx.scalar.read(0), 0xABCD);
    }

    #[test]
    fn test_vector_memory_helpers() {
        let mut tile = make_tile();

        let data = [1, 2, 3, 4, 5, 6, 7, 8];
        MemoryUnit::write_vector_to_memory(&mut tile, 0x300, data, None);

        let read_back = MemoryUnit::read_vector_from_memory(&tile, 0x300, None);
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
        MemoryUnit::write_vector_to_memory(&mut tile, 0x400, test_data, None);

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

        MemoryUnit::execute(&op, &mut ctx, &mut tile, None);
        ctx.flush_pending_writes();

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

        MemoryUnit::execute(&op, &mut ctx, &mut tile, None);

        // Verify all 8 lanes were stored correctly
        let stored = MemoryUnit::read_vector_from_memory(&tile, 0x500, None);
        assert_eq!(stored, test_data);
    }

    #[test]
    fn test_vector256_load_with_post_modify() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        let test_data = [1, 2, 3, 4, 5, 6, 7, 8];
        MemoryUnit::write_vector_to_memory(&mut tile, 0x600, test_data, None);
        ctx.pointer.write(0, 0x600);

        // Load with post-modify: v0 = [p0], p0 += 32
        let op = SlotOp::new(
            SlotIndex::Load,
            Operation::Load {
                width: MemWidth::Vector256,
                post_modify: PostModify::Immediate(32), // 256 bits = 32 bytes
            },
        )
        .with_post_modify(PostModify::Immediate(32))
        .with_dest(Operand::VectorReg(0))
        .with_source(Operand::PointerReg(0));

        MemoryUnit::execute(&op, &mut ctx, &mut tile, None);
        ctx.flush_pending_writes();

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
        MemoryUnit::write_vector_to_memory(&mut tile, 0x700, test_data, None);
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

        assert!(MemoryUnit::execute(&op, &mut ctx, &mut tile, None));
        ctx.flush_pending_writes();
        assert_eq!(ctx.vector.read(2), test_data);
        assert_eq!(ctx.pointer.read(0), 0x700); // No post-modify
    }

    #[test]
    fn test_vector_load_a_with_post_modify() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        let test_data = [1, 2, 3, 4, 5, 6, 7, 8];
        MemoryUnit::write_vector_to_memory(&mut tile, 0x800, test_data, None);
        ctx.pointer.write(1, 0x800);

        // VLDA: v0 = [p1], p1 += 32
        let op = SlotOp::new(
            SlotIndex::Load,
            Operation::VectorLoadA {
                post_modify: PostModify::Immediate(32),
            },
        )
        .with_post_modify(PostModify::Immediate(32))
        .with_dest(Operand::VectorReg(0))
        .with_source(Operand::PointerReg(1));

        MemoryUnit::execute(&op, &mut ctx, &mut tile, None);
        ctx.flush_pending_writes();
        assert_eq!(ctx.vector.read(0), test_data);
        assert_eq!(ctx.pointer.read(1), 0x820); // 0x800 + 32
    }

    #[test]
    fn test_vector_load_b() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        let test_data = [0xDEADBEEF, 0xCAFEBABE, 0x12345678, 0x9ABCDEF0,
                         0x0F0F0F0F, 0xF0F0F0F0, 0xAAAA5555, 0x5555AAAA];
        MemoryUnit::write_vector_to_memory(&mut tile, 0x900, test_data, None);
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

        assert!(MemoryUnit::execute(&op, &mut ctx, &mut tile, None));
        ctx.flush_pending_writes();
        assert_eq!(ctx.vector.read(3), test_data);
    }

    #[test]
    fn test_vector_load_b_with_modifier_register() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        let test_data = [10, 20, 30, 40, 50, 60, 70, 80];
        MemoryUnit::write_vector_to_memory(&mut tile, 0xA00, test_data, None);
        ctx.pointer.write(5, 0xA00);
        ctx.modifier.write(2, 64); // m2 = 64

        // VLDB: v1 = [p5], p5 += m2
        let op = SlotOp::new(
            SlotIndex::Load,
            Operation::VectorLoadB {
                post_modify: PostModify::Register(2),
            },
        )
        .with_post_modify(PostModify::Register(2))
        .with_dest(Operand::VectorReg(1))
        .with_source(Operand::PointerReg(5));

        MemoryUnit::execute(&op, &mut ctx, &mut tile, None);
        ctx.flush_pending_writes();
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

        assert!(MemoryUnit::execute(&op, &mut ctx, &mut tile, None));

        let stored = MemoryUnit::read_vector_from_memory(&tile, 0xB00, None);
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
        .with_post_modify(PostModify::Immediate(32))
        .with_dest(Operand::VectorReg(5))
        .with_source(Operand::PointerReg(3));

        MemoryUnit::execute(&op, &mut ctx, &mut tile, None);

        let stored = MemoryUnit::read_vector_from_memory(&tile, 0xC00, None);
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

        assert!(MemoryUnit::execute(&op, &mut ctx, &mut tile, None));
        ctx.flush_pending_writes();

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

        MemoryUnit::execute(&op, &mut ctx, &mut tile, None);
        ctx.flush_pending_writes();

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
        .with_post_modify(PostModify::Immediate(16))
        .with_dest(Operand::VectorReg(0))
        .with_source(Operand::PointerReg(2));

        MemoryUnit::execute(&op, &mut ctx, &mut tile, None);
        ctx.flush_pending_writes();

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

    // --- Cross-tile memory tests ---

    #[test]
    fn test_neighbor_memory_build_and_read() {
        // Create a device with tiles. Tile(1,2) is the executing core.
        // Neighbor mapping:
        //   Quadrant 1 (West)  = tile(0, 2)
        //   Quadrant 2 (North) = tile(1, 3)
        //   Quadrant 3 (East)  = tile(2, 2)
        let mut device = crate::device::DeviceState::new_npu1();

        // Write known data into west neighbor's memory
        if let Some(west) = device.tile_mut(0, 2) {
            west.data_memory_mut()[0x100..0x104].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        }

        // Write known data into north neighbor's memory
        if let Some(north) = device.tile_mut(1, 3) {
            north.data_memory_mut()[0x200..0x204].copy_from_slice(&0xCAFE_BABEu32.to_le_bytes());
        }

        // Write known data into east neighbor's memory
        if let Some(east) = device.tile_mut(2, 2) {
            east.data_memory_mut()[0x300..0x304].copy_from_slice(&0x1234_5678u32.to_le_bytes());
        }

        // Build NeighborMemory for tile(1,2) and load snapshots
        let mut nbr = NeighborMemory::new(1, 2);
        nbr.ensure_snapshot(1, &device); // West
        nbr.ensure_snapshot(2, &device); // North
        nbr.ensure_snapshot(3, &device); // East

        // Verify reads from each quadrant
        let west_mem = nbr.get_memory(1).expect("West neighbor should exist");
        assert_eq!(
            u32::from_le_bytes([west_mem[0x100], west_mem[0x101], west_mem[0x102], west_mem[0x103]]),
            0xDEAD_BEEF
        );

        let north_mem = nbr.get_memory(2).expect("North neighbor should exist");
        assert_eq!(
            u32::from_le_bytes([north_mem[0x200], north_mem[0x201], north_mem[0x202], north_mem[0x203]]),
            0xCAFE_BABE
        );

        let east_mem = nbr.get_memory(3).expect("East neighbor should exist");
        assert_eq!(
            u32::from_le_bytes([east_mem[0x300], east_mem[0x301], east_mem[0x302], east_mem[0x303]]),
            0x1234_5678
        );
    }

    #[test]
    fn test_neighbor_memory_buffer_write_and_apply() {
        let mut device = crate::device::DeviceState::new_npu1();

        // Build NeighborMemory for tile(1,2)
        let mut nbr = NeighborMemory::new(1, 2);
        nbr.ensure_snapshot(1, &device);

        // Buffer a write to west neighbor at offset 0x400
        let write_data = 0xAAAA_BBBBu32.to_le_bytes();
        nbr.buffer_write(1, 0x400, &write_data);

        assert!(nbr.has_pending_writes());

        // Apply writes to the device
        nbr.apply_writes(&mut device);

        // Verify the write landed in the west neighbor tile
        let west = device.tile(0, 2).unwrap();
        let mem = west.data_memory();
        let value = u32::from_le_bytes([mem[0x400], mem[0x401], mem[0x402], mem[0x403]]);
        assert_eq!(value, 0xAAAA_BBBB);
    }

    #[test]
    fn test_cross_tile_scalar_load() {
        // Test that a load with address in quadrant 1 (0x1xxxx) reads from
        // the west neighbor's memory snapshot.
        let mut device = crate::device::DeviceState::new_npu1();

        // Write test data to west neighbor (tile 0,2)
        if let Some(west) = device.tile_mut(0, 2) {
            west.write_data_u32(0x100, 0xFEED_FACE);
        }

        // Build neighbor snapshot for tile(1,2)
        let mut nbr = NeighborMemory::new(1, 2);
        nbr.ensure_snapshot(1, &device);

        // Read from quadrant 1 address: 0x10100 = west neighbor, offset 0x100
        let tile = Tile::compute(1, 2);
        let value = MemoryUnit::read_memory(&tile, 0x10100, MemWidth::Word, Some(&nbr));
        assert_eq!(value, 0xFEED_FACE);
    }

    #[test]
    fn test_cross_tile_vector_load() {
        let mut device = crate::device::DeviceState::new_npu1();

        // Write vector data to north neighbor (tile 1,3)
        let test_data = [10u32, 20, 30, 40, 50, 60, 70, 80];
        if let Some(north) = device.tile_mut(1, 3) {
            let mem = north.data_memory_mut();
            for (i, &val) in test_data.iter().enumerate() {
                let offset = 0x200 + i * 4;
                let bytes = val.to_le_bytes();
                mem[offset..offset + 4].copy_from_slice(&bytes);
            }
        }

        // Build neighbor snapshot
        let mut nbr = NeighborMemory::new(1, 2);
        nbr.ensure_snapshot(2, &device); // North

        // Read vector from quadrant 2: 0x20200 = north neighbor, offset 0x200
        let tile = Tile::compute(1, 2);
        let result = MemoryUnit::read_vector_from_memory(&tile, 0x20200, Some(&nbr));
        assert_eq!(result, test_data);
    }

    #[test]
    fn test_cross_tile_scalar_store_buffered() {
        // Cross-tile stores are buffered, not immediately written.
        let mut device = crate::device::DeviceState::new_npu1();

        let mut nbr = NeighborMemory::new(1, 2);
        nbr.ensure_snapshot(3, &device); // East

        // Write to quadrant 3 (east) address 0x30400
        let mut tile = Tile::compute(1, 2);
        MemoryUnit::write_memory(&mut tile, 0x30400, 0xBAAD_F00D, MemWidth::Word, Some(&mut nbr));

        // Should be buffered, not yet in the east neighbor
        let east = device.tile(2, 2).unwrap();
        let mem = east.data_memory();
        let value = u32::from_le_bytes([mem[0x400], mem[0x401], mem[0x402], mem[0x403]]);
        assert_eq!(value, 0, "Write should be buffered, not applied yet");

        // Apply writes
        assert!(nbr.has_pending_writes());
        nbr.apply_writes(&mut device);

        // Now it should be visible
        let east = device.tile(2, 2).unwrap();
        let mem = east.data_memory();
        let value = u32::from_le_bytes([mem[0x400], mem[0x401], mem[0x402], mem[0x403]]);
        assert_eq!(value, 0xBAAD_F00D);
    }

    #[test]
    fn test_cross_tile_vector_store_buffered() {
        let mut device = crate::device::DeviceState::new_npu1();

        let mut nbr = NeighborMemory::new(1, 2);
        nbr.ensure_snapshot(1, &device); // West

        // Write vector to quadrant 1 (west) address 0x10800
        let test_data = [1u32, 2, 3, 4, 5, 6, 7, 8];
        let mut tile = Tile::compute(1, 2);
        MemoryUnit::write_vector_to_memory(&mut tile, 0x10800, test_data, Some(&mut nbr));

        // Apply and verify
        nbr.apply_writes(&mut device);

        let west = device.tile(0, 2).unwrap();
        let mem = west.data_memory();
        for (i, &expected) in test_data.iter().enumerate() {
            let offset = 0x800 + i * 4;
            let actual = u32::from_le_bytes([mem[offset], mem[offset+1], mem[offset+2], mem[offset+3]]);
            assert_eq!(actual, expected, "Vector word {} mismatch", i);
        }
    }

    #[test]
    fn test_quadrant_0_fast_path_unchanged() {
        // Verify that quadrant 0 (local memory) works exactly as before,
        // even when NeighborMemory is Some.
        let mut device = crate::device::DeviceState::new_npu1();
        let mut nbr = NeighborMemory::new(1, 2);
        nbr.ensure_snapshot(1, &device);

        let mut tile = make_tile();
        tile.write_data_u32(0x100, 0x42424242);

        // Read from local memory (quadrant 0) with neighbors present
        let value = MemoryUnit::read_memory(&tile, 0x100, MemWidth::Word, Some(&nbr));
        assert_eq!(value, 0x42424242);

        // Write to local memory (quadrant 0) with neighbors present
        MemoryUnit::write_memory(&mut tile, 0x200, 0x99887766, MemWidth::Word, Some(&mut nbr));
        assert_eq!(tile.read_data_u32(0x200), Some(0x99887766));

        // No cross-tile writes should have been buffered
        assert!(!nbr.has_pending_writes());
    }

    #[test]
    fn test_edge_tile_no_west_neighbor() {
        // Tile(0,2) has no west neighbor (col 0).
        // Cross-tile read should return 0.
        let device = crate::device::DeviceState::new_npu1();
        let mut nbr = NeighborMemory::new(0, 2);
        nbr.ensure_snapshot(1, &device); // West -- doesn't exist

        let tile = Tile::compute(0, 2);
        let value = MemoryUnit::read_memory(&tile, 0x10100, MemWidth::Word, Some(&nbr));
        assert_eq!(value, 0, "Read from non-existent west neighbor should return 0");
    }

    #[test]
    fn test_load_latency_local_vs_cross_tile() {
        use crate::interpreter::timing::CROSS_TILE_LATENCY;

        // Local memory (quadrant 0): base latency only
        assert_eq!(load_latency_for_address(0x0000), LATENCY_MEMORY as u64);
        assert_eq!(load_latency_for_address(0x0FFFF), LATENCY_MEMORY as u64);

        // West neighbor (quadrant 1): base + cross-tile
        let expected_cross = LATENCY_MEMORY as u64 + CROSS_TILE_LATENCY as u64;
        assert_eq!(load_latency_for_address(0x10000), expected_cross);
        assert_eq!(load_latency_for_address(0x1FFFF), expected_cross);

        // North neighbor (quadrant 2): base + cross-tile
        assert_eq!(load_latency_for_address(0x20000), expected_cross);

        // East neighbor (quadrant 3): base + cross-tile
        assert_eq!(load_latency_for_address(0x30000), expected_cross);

        // High linker address (>0x3FFFF) maps to local (quadrant 0)
        assert_eq!(load_latency_for_address(0x70440), LATENCY_MEMORY as u64);
    }

    #[test]
    fn test_linker_high_bits_masked_to_local() {
        // Linker-assigned addresses like 0x70440 have high bits beyond the
        // 18-bit AGU range. They should be masked to quadrant 0 local access.
        // 0x70440 & 0x3FFFF = 0x30440 -> quadrant 3, offset 0x0440
        // But actually, linker addresses are typically resolved by ELF loading
        // which subtracts the base (0x70000). The AGU only sees 20-bit addresses.
        //
        // For addresses where bits above 17 are set but below 20, they map
        // correctly to quadrants 0-3. Addresses above 0x3FFFF wrap via the
        // mask to whatever quadrant the lower bits indicate.
        let mut tile = make_tile();
        tile.write_data_u32(0x440, 0xABCDABCD);

        // 0x00440 = quadrant 0, offset 0x440 -> local memory
        let value = MemoryUnit::read_memory(&tile, 0x00440, MemWidth::Word, None);
        assert_eq!(value, 0xABCDABCD);
    }
}
