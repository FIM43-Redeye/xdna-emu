//! Memory unit execution.
//!
//! Handles load/store operations between registers and tile memory.
//!
//! Memory operations require tile access for actual reads/writes, so they
//! are handled here rather than in semantic dispatch.
//!
//! # Addressing
//!
//! AIE2 uses pointer registers (p0-p7) for addressing with optional
//! post-modify (add immediate or modifier register after access).
//!
//! The AGU (Address Generation Unit) generates 20-bit addresses in the
//! range 0x40000-0x7FFFF (256KB across 4 cardinal directions):
//! - 0x40000-0x4FFFF: South neighbor (same col, row-1) [CardDir 4]
//! - 0x50000-0x5FFFF: West neighbor (col-1, same row)  [CardDir 5]
//! - 0x60000-0x6FFFF: North neighbor (same col, row+1) [CardDir 6]
//! - 0x70000-0x7FFFF: East = LOCAL memory (AIE2)       [CardDir 7]
//!
//! The cardinal direction is `CardDir = address / MEMORY_SIZE`. On AIE2
//! (IsCheckerBoard=0), East is always the local tile's own memory. On
//! checkerboard architectures (AIE1), the local direction depends on
//! row parity.
//!
//! Source: aie-rt `_XAie_GetTargetTileLoc()` (xaie_elfloader.c:124-183)
//!
//! # Memory Layout
//!
//! - **Data Memory**: 64KB at 0x70000-0x7FFFF in core view (compute tile)
//! - **Program Memory**: 16KB (read-only from core)

use crate::device::tile::Tile;
use crate::interpreter::bundle::{ElementType, MemWidth, Operand, PostModify, SlotIndex, SlotOp};
use crate::interpreter::state::ExecutionContext;
use crate::interpreter::timing::{LATENCY_MEMORY, CROSS_TILE_LATENCY, MemoryQuadrant};
use crate::tablegen::SemanticOp;

/// Offset mask for extracting the local address within a tile's memory.
/// `address & OFFSET_MASK` gives the byte offset within the target tile.
const OFFSET_MASK: u32 = crate::arch::compute::MEMORY_SIZE as u32 - 1;

/// Decode a data memory address into its cardinal direction and local offset.
///
/// AIE2 cores use a 20-bit data address space (0x40000-0x7FFFF):
///   - `CardDir = address / MEMORY_SIZE` selects the target tile
///   - `address & OFFSET_MASK` is the byte offset within that tile
///
/// CardDir 4=South, 5=West, 6=North, 7=East(local on AIE2).
///
/// The ELF linker places data at 0x70000 because that IS the hardware
/// address for the core's own data memory (CardDir 7 = East = local).
///
/// Source: aie-rt `_XAie_GetTargetTileLoc()` (xaie_elfloader.c:124-183)
#[inline]
fn decode_data_address(addr: u32) -> (MemoryQuadrant, usize) {
    let offset = (addr & OFFSET_MASK) as usize;
    (MemoryQuadrant::from_address(addr), offset)
}

/// Compute the total load latency for a data memory address.
///
/// Local memory uses the base LATENCY_MEMORY (7 cycles).
/// Cross-tile accesses add CROSS_TILE_LATENCY (4 cycles) per hop
/// to model the routing delay through the stream switch.
#[inline]
fn load_latency_for_address(addr: u32) -> u64 {
    let (quadrant, _) = decode_data_address(addr);
    let base = LATENCY_MEMORY as u64;
    if quadrant == MemoryQuadrant::Local {
        base
    } else {
        base + CROSS_TILE_LATENCY as u64
    }
}

/// Read-only snapshots of neighbor tile data memory for cross-tile access.
///
/// Built lazily before each core step. Neighbor memory is only cloned on
/// the first cross-tile access, so cores that only touch local memory
/// pay zero allocation cost.
///
/// Cross-tile writes are buffered and applied after the core step completes,
/// matching hardware behavior where cross-tile writes have higher latency
/// and become visible on the next cycle.
///
/// # Cardinal Direction Mapping
///
/// | CardDir | Direction | Neighbor offset        | AIE2 behavior |
/// |---------|-----------|------------------------|---------------|
/// | 4       | South     | (col, row-1)           | Cross-tile    |
/// | 5       | West      | (col-1, row)           | Cross-tile    |
/// | 6       | North     | (col, row+1)           | Cross-tile    |
/// | 7       | East      | (col+1, row) or LOCAL  | Local (AIE2)  |
///
/// On AIE2 (IsCheckerBoard=0), East is always local -- `decode_data_address`
/// maps CardDir 7 to `MemoryQuadrant::Local`, so the East slot is unused.
/// On checkerboard architectures (AIE1), East may be a real cross-tile
/// neighbor depending on row parity.
pub struct NeighborMemory {
    /// Source tile coordinates (needed for resolving neighbors).
    col: usize,
    row: usize,

    /// Neighbor data memory snapshots indexed by cardinal direction:
    /// 0=South, 1=West, 2=North, 3=East.
    /// None until first access (lazy clone). Inner None if neighbor doesn't exist.
    snapshots: [Option<Option<Vec<u8>>>; 4],

    /// Buffered cross-tile writes: (direction, offset_within_tile, data).
    /// Applied after core step completes.
    pub pending_writes: Vec<(MemoryQuadrant, usize, Vec<u8>)>,
}

/// Map a cardinal direction to its snapshot array index.
///
/// Returns None for `Local` (no snapshot needed -- use own tile memory).
fn dir_index(dir: MemoryQuadrant) -> Option<usize> {
    match dir {
        MemoryQuadrant::South => Some(0),
        MemoryQuadrant::West => Some(1),
        MemoryQuadrant::North => Some(2),
        MemoryQuadrant::East => Some(3),
        MemoryQuadrant::Local => None,
    }
}

impl NeighborMemory {
    /// Create a new NeighborMemory for the core at (col, row).
    ///
    /// Does NOT clone any neighbor memory yet -- snapshots are lazy.
    pub fn new(col: usize, row: usize) -> Self {
        Self {
            col,
            row,
            snapshots: [None, None, None, None],
            pending_writes: Vec::new(),
        }
    }

    /// Resolve a cardinal direction to the neighbor tile coordinates.
    ///
    /// Returns None if the neighbor is outside the array bounds or if
    /// the direction is `Local`.
    fn neighbor_coords(&self, dir: MemoryQuadrant) -> Option<(usize, usize)> {
        match dir {
            MemoryQuadrant::South => {
                if self.row > 0 { Some((self.col, self.row - 1)) } else { None }
            }
            MemoryQuadrant::West => {
                if self.col > 0 { Some((self.col - 1, self.row)) } else { None }
            }
            MemoryQuadrant::North => Some((self.col, self.row + 1)),
            MemoryQuadrant::East => Some((self.col + 1, self.row)),
            MemoryQuadrant::Local => None,
        }
    }

    /// Lazily snapshot a neighbor tile's data memory.
    ///
    /// Called before execution for each direction that might be accessed.
    /// The `device` reference is only needed for this initial clone.
    pub fn ensure_snapshot(&mut self, dir: MemoryQuadrant, device: &crate::device::DeviceState) {
        let idx = match dir_index(dir) {
            Some(i) => i,
            None => return, // Local -- no snapshot needed
        };
        if self.snapshots[idx].is_some() {
            return; // Already loaded
        }

        let snapshot = self.neighbor_coords(dir)
            .and_then(|(c, r)| device.tile(c, r))
            .map(|tile| tile.data_memory().to_vec());

        self.snapshots[idx] = Some(snapshot);
    }

    /// Get a reference to a neighbor's data memory snapshot.
    ///
    /// Returns None if the neighbor doesn't exist or hasn't been snapshotted.
    pub fn get_memory(&self, dir: MemoryQuadrant) -> Option<&[u8]> {
        let idx = dir_index(dir)?;
        self.snapshots[idx]
            .as_ref()
            .and_then(|opt| opt.as_deref())
    }

    /// Buffer a cross-tile write for deferred application.
    pub fn buffer_write(&mut self, dir: MemoryQuadrant, offset: usize, data: &[u8]) {
        self.pending_writes.push((dir, offset, data.to_vec()));
    }

    /// Apply all buffered cross-tile writes to the device.
    ///
    /// Resolves each direction to the target tile and writes the data.
    /// Called after the core step completes.
    pub fn apply_writes(self, device: &mut crate::device::DeviceState) {
        let col = self.col;
        let row = self.row;
        for (dir, offset, data) in self.pending_writes {
            let coords = match dir {
                MemoryQuadrant::South => if row > 0 { Some((col, row - 1)) } else { None },
                MemoryQuadrant::West => if col > 0 { Some((col - 1, row)) } else { None },
                MemoryQuadrant::North => Some((col, row + 1)),
                MemoryQuadrant::East => Some((col + 1, row)),
                MemoryQuadrant::Local => None,
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
    /// The `neighbors` parameter enables cross-tile memory access.
    /// Pass `None` for local-only access (unit tests, simple mode).
    ///
    /// Returns `true` if the operation was handled, `false` if not a memory op.
    pub fn execute(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        tile: &mut Tile,
        neighbors: Option<&mut NeighborMemory>,
    ) -> bool {
        // PostModify comes from op.post_modify (populated from the AG field
        // during decode).
        let pm = &op.post_modify;

        match op.semantic {
            Some(SemanticOp::Load) if !op.is_vector => {
                Self::execute_load(op, ctx, tile, op.mem_width, pm, neighbors);
                true
            }

            Some(SemanticOp::Store) if !op.is_vector => {
                Self::execute_store(op, ctx, tile, op.mem_width, pm, neighbors);
                true
            }

            Some(SemanticOp::Load) if op.slot == SlotIndex::LoadA => {
                Self::execute_vector_load_a(op, ctx, tile, pm, neighbors);
                true
            }

            Some(SemanticOp::Load) if op.slot == SlotIndex::LoadB => {
                Self::execute_vector_load_b(op, ctx, tile, pm, neighbors);
                true
            }

            Some(SemanticOp::Unpack) => {
                let from_type = op.from_type.unwrap_or(ElementType::Int32);
                let to_type = op.element_type.unwrap_or(ElementType::Int32);
                Self::execute_vector_load_unpack(op, ctx, tile, from_type, to_type, pm);
                true
            }

            Some(SemanticOp::Store) if op.is_vector => {
                Self::execute_vector_store(op, ctx, tile, pm, neighbors);
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
        neighbors: Option<&mut NeighborMemory>,
    ) {
        // Get address from source operand
        let addr = Self::get_address(op, ctx);
        let latency = load_latency_for_address(addr);

        // Track bank access for conflict detection (local memory only)
        let (quadrant, local_offset) = decode_data_address(addr);
        if quadrant == MemoryQuadrant::Local {
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
                        "[WATCH-LD] pc=0x{:03X} cycle={} elem={} addr=0x{:05X} value=0x{:08X} dest={:?}",
                        ctx.pc(), ctx.cycles, elem, masked, value as u32, op.dest
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
        if quadrant == MemoryQuadrant::Local {
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
            // AIE2 partial-word stores (st.s8, st.u8, st.s16, st.u16) use a
            // read-modify-write pipeline. The DATA register is read 7 cycles
            // after issue (II_STHB operand latency = 7 in AIE2Schedule.td).
            // The address is computed at issue time; only the data read is late.
            // This allows the compiler to schedule computations between the
            // store instruction and the data read cycle.
            if width.is_partial_word() {
                let source = Self::get_store_source_operand(op);
                ctx.queue_pending_store(addr, source, width);
            } else {
                let value = Self::get_store_value(op, ctx, width);
                Self::write_memory(tile, addr, value, width, neighbors);
            }
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
        if quadrant == MemoryQuadrant::Local {
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
        if quadrant == MemoryQuadrant::Local {
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
        if quadrant == MemoryQuadrant::Local {
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
            other => {
                log::warn!("[MEMORY] get_address: unexpected base operand {:?}, defaulting to 0", other);
                0
            }
        });

        let offset = op.sources.get(1).map_or(0, |src| match src {
            Operand::Immediate(v) => (*v as i32 * 4) as u32, // Scale by word size
            Operand::ScalarReg(r) => ctx.scalar_read(*r).wrapping_mul(4),
            // Modifier registers contain byte offsets (set via mov dj0/m0, rN)
            Operand::ModifierReg(r) => ctx.modifier_read(*r),
            other => {
                log::warn!("[MEMORY] get_address: unexpected offset operand {:?}, defaulting to 0", other);
                0
            }
        });

        base_addr.wrapping_add(offset)
    }

    /// Get store address.
    ///
    /// Handles three operand layouts:
    /// 1. Memory operand: `Memory { base, offset }` -- pre-scaled byte offset
    /// 2. Test/legacy: sources[0]=ptr (no offset)
    /// 3. Kernel layout: sources[0]=value, sources[1]=ptr, sources[2]=index
    ///    The index is a modifier register (dj/m) containing a byte offset
    ///    for indexed addressing (`st.s16 rN, [pM, djK]`).
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

        // Kernel layout: sources[0]=value, sources[1]=pointer, sources[2]=index
        let base_addr = op.sources.get(1).map_or(0, |src| match src {
            Operand::PointerReg(r) => ctx.pointer_read(*r),
            Operand::ScalarReg(r) => ctx.scalar_read(*r),
            Operand::Immediate(v) => *v as u32,
            other => {
                log::warn!("[MEMORY] get_store_address: unexpected base operand {:?}, defaulting to 0", other);
                0
            }
        });

        // sources[2] is the index offset: modifier registers contain byte
        // offsets (from mov dj/m, rN), matching the load path in get_address.
        let offset = op.sources.get(2).map_or(0, |src| match src {
            Operand::ModifierReg(r) => ctx.modifier_read(*r),
            Operand::Immediate(v) => (*v as i32 * 4) as u32,
            Operand::ScalarReg(r) => ctx.scalar_read(*r).wrapping_mul(4),
            other => {
                log::warn!("[MEMORY] get_store_address: unexpected offset operand {:?}, defaulting to 0", other);
                0
            }
        });

        base_addr.wrapping_add(offset)
    }

    /// Get value to store.
    ///
    /// For TableGen-decoded instructions (encoding_name is Some), LLVM's
    /// InOperandList invariant guarantees sources[0] is always the value.
    /// This is true across every AIE2 store family: ST, ST_S8, VST, spills.
    ///
    /// For hand-constructed test SlotOps (encoding_name is None), falls back
    /// to a legacy heuristic that checks operand types.
    fn get_store_value(op: &SlotOp, ctx: &ExecutionContext, width: MemWidth) -> u64 {
        let operand = if op.encoding_name.is_some() {
            // TableGen-decoded: sources[0] is always the value (LLVM convention).
            // The InOperandList for every AIE2 store puts the value register
            // class (mSclSt, eR, mWs, etc.) first, followed by address/offset.
            op.sources.first()
        } else {
            // Hand-constructed SlotOp (tests): use legacy layout detection.
            // If sources[0] is an address, value is in dest.
            op.sources.first().and_then(|first| {
                match first {
                    Operand::PointerReg(_) | Operand::Memory { .. } => op.dest.as_ref(),
                    _ => Some(first),
                }
            }).or(op.dest.as_ref())
        };

        Self::read_store_operand(operand, ctx, width)
    }

    /// Read a store value from an operand reference.
    /// Extract the source register operand for a store instruction.
    ///
    /// Uses the same logic as `get_store_value` but returns the Operand
    /// instead of reading it, for use with deferred partial-word stores.
    fn get_store_source_operand(op: &SlotOp) -> Operand {
        let operand = if op.encoding_name.is_some() {
            op.sources.first().cloned()
        } else {
            op.sources.first().and_then(|first| {
                match first {
                    Operand::PointerReg(_) | Operand::Memory { .. } => op.dest.clone(),
                    _ => Some(first.clone()),
                }
            }).or_else(|| op.dest.clone())
        };
        operand.unwrap_or(Operand::ScalarReg(0))
    }

    /// Read a register value for a deferred partial-word store.
    ///
    /// Called from the cycle-accurate execution loop when a pending store's
    /// data-read cycle arrives. Uses the same register read path as immediate
    /// stores, including VLIW forwarding.
    pub fn read_store_register(operand: &Operand, ctx: &ExecutionContext, width: MemWidth) -> u64 {
        Self::read_store_operand(Some(operand), ctx, width)
    }

    fn read_store_operand(operand: Option<&Operand>, ctx: &ExecutionContext, width: MemWidth) -> u64 {
        operand.map_or(0, |src| match src {
            Operand::ScalarReg(r) => ctx.scalar_read(*r) as u64,
            Operand::PointerReg(r) => ctx.pointer_read(*r) as u64,
            Operand::ModifierReg(r) => ctx.modifier_read(*r) as u64,
            Operand::VectorReg(r) => {
                let vec = ctx.vector.read(*r);
                if width == MemWidth::Vector256 {
                    ((vec[1] as u64) << 32) | (vec[0] as u64)
                } else {
                    vec[0] as u64
                }
            }
            Operand::Immediate(v) => *v as u64,
            _ => 0,
        })
    }

    /// Read from tile memory with cross-tile cardinal direction routing.
    ///
    /// CardDir = address / MEMORY_SIZE selects the target tile:
    /// - 4: South neighbor   - 5: West neighbor
    /// - 6: North neighbor   - 7: East = Local (AIE2)
    ///
    /// Offset within the target tile is `address & OFFSET_MASK`.
    fn read_memory(tile: &Tile, addr: u32, width: MemWidth, neighbors: Option<&NeighborMemory>) -> u64 {
        let (quadrant, offset) = decode_data_address(addr);

        // Select memory source based on quadrant
        let mem: &[u8] = if quadrant == MemoryQuadrant::Local || neighbors.is_none() {
            // Fast path: local tile memory
            tile.data_memory()
        } else if let Some(neighbor_mem) = neighbors.and_then(|n| n.get_memory(quadrant)) {
            neighbor_mem
        } else {
            // Neighbor doesn't exist (edge of array) - return zero
            log::trace!("[LOAD] cross-tile read to non-existent neighbor {:?}", quadrant);
            return 0;
        };

        Self::read_from_slice(mem, offset, width)
    }

    /// Read a value from a memory slice at the given offset.
    fn read_from_slice(mem: &[u8], addr: usize, width: MemWidth) -> u64 {
        if addr >= mem.len() {
            log::warn!(
                "[MEMORY] OOB read: addr=0x{:X} width={:?} mem_size=0x{:X}",
                addr, width, mem.len()
            );
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

    /// Write to tile memory with cross-tile cardinal direction routing.
    ///
    /// Cross-tile writes (South/West/North/East) are buffered in the
    /// NeighborMemory for deferred application after the core step completes.
    pub fn write_memory(tile: &mut Tile, addr: u32, value: u64, width: MemWidth, neighbors: Option<&mut NeighborMemory>) {
        let (quadrant, offset) = decode_data_address(addr);

        if quadrant != MemoryQuadrant::Local {
            if let Some(nbr) = neighbors {
                // Cross-tile write: serialize value and buffer it
                let data = Self::serialize_value(value, width);
                nbr.buffer_write(quadrant, offset, &data);
                return;
            }
            // No neighbor context -- fall through to local write (legacy behavior)
        }

        // Local tile write (Local direction or no neighbor context)
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
    /// memory pipeline. Cross-tile accesses (South/West/North) incur additional
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
                    use crate::interpreter::state::SP_PTR_INDEX;
                    if reg == SP_PTR_INDEX {
                        let sp = ctx.sp();
                        ctx.set_sp(sp.wrapping_add(*imm as i32 as u32));
                    } else {
                        ctx.pointer.add(reg, *imm as i32);
                    }
                }
                PostModify::Register(m) => {
                    use crate::interpreter::state::SP_PTR_INDEX;
                    let modifier = ctx.modifier_read(*m) as i32;
                    if reg == SP_PTR_INDEX {
                        let sp = ctx.sp();
                        ctx.set_sp(sp.wrapping_add(modifier as u32));
                    } else {
                        ctx.pointer.add(reg, modifier);
                    }
                }
            }
        }
    }

    /// Read a vector from memory (256 bits = 32 bytes) with cross-tile routing.
    pub fn read_vector_from_memory(tile: &Tile, addr: u32, neighbors: Option<&NeighborMemory>) -> [u32; 8] {
        let (quadrant, offset) = decode_data_address(addr);

        // Select memory source
        let mem: &[u8] = if quadrant == MemoryQuadrant::Local || neighbors.is_none() {
            tile.data_memory()
        } else if let Some(neighbor_mem) = neighbors.and_then(|n| n.get_memory(quadrant)) {
            neighbor_mem
        } else {
            log::trace!("[VLOAD] cross-tile read to non-existent neighbor {:?}", quadrant);
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

        if quadrant != MemoryQuadrant::Local {
            if let Some(nbr) = neighbors {
                // Cross-tile vector write: serialize all 8 words and buffer
                let mut data = Vec::with_capacity(32);
                for word in &value {
                    data.extend_from_slice(&word.to_le_bytes());
                }
                nbr.buffer_write(quadrant, offset, &data);
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
    use crate::interpreter::bundle::{ElementType, SlotIndex};
    use crate::interpreter::state::SP_PTR_INDEX;
    use crate::tablegen::SemanticOp;

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
        let op = SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load)
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
        let op = SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load)
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
        let op = SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load)
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
        let op = SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load)
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
        let op = SlotOp::from_semantic(SlotIndex::Store, SemanticOp::Store)
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

        let op = SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load)
            .with_mem_width(MemWidth::Byte)
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

        let op = SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load)
            .with_mem_width(MemWidth::HalfWord)
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
        let op = SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load)
            .with_mem_width(MemWidth::Vector256)
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
        let op = SlotOp::from_semantic(SlotIndex::Store, SemanticOp::Store)
            .with_mem_width(MemWidth::Vector256)
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
        let op = SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load)
            .with_mem_width(MemWidth::Vector256)
            .with_post_modify(PostModify::Immediate(32)) // 256 bits = 32 bytes
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
        let op = SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load)
            .as_vector(ElementType::Int32)
            .with_mem_width(MemWidth::Vector256)
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
        let op = SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load)
            .as_vector(ElementType::Int32)
            .with_mem_width(MemWidth::Vector256)
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
        let op = SlotOp::from_semantic(SlotIndex::LoadB, SemanticOp::Load)
            .as_vector(ElementType::Int32)
            .with_mem_width(MemWidth::Vector256)
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
        let op = SlotOp::from_semantic(SlotIndex::LoadB, SemanticOp::Load)
            .as_vector(ElementType::Int32)
            .with_mem_width(MemWidth::Vector256)
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
        let op = SlotOp::from_semantic(SlotIndex::Store, SemanticOp::Store)
            .as_vector(ElementType::Int32)
            .with_mem_width(MemWidth::Vector256)
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
        let op = SlotOp::from_semantic(SlotIndex::Store, SemanticOp::Store)
            .as_vector(ElementType::Int32)
            .with_mem_width(MemWidth::Vector256)
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
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Write 8 bytes of int8 data
        let mem = tile.data_memory_mut();
        for i in 0..8 {
            mem[0xD00 + i] = (i + 1) as u8; // 1, 2, 3, 4, 5, 6, 7, 8
        }
        ctx.pointer.write(0, 0xD00);

        // VLDB_UNPACK: expand int8 to int32
        let mut op = SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Unpack)
            .with_dest(Operand::VectorReg(6))
            .with_source(Operand::PointerReg(0));
        op.is_vector = true;
        op.from_type = Some(ElementType::UInt8);
        op.element_type = Some(ElementType::UInt32);

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
        let mut op = SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Unpack)
            .with_dest(Operand::VectorReg(7))
            .with_source(Operand::PointerReg(1));
        op.is_vector = true;
        op.from_type = Some(ElementType::Int8);
        op.element_type = Some(ElementType::Int32);

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
        let mut op = SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Unpack)
            .with_post_modify(PostModify::Immediate(16)) // Read 16 bytes
            .with_dest(Operand::VectorReg(0))
            .with_source(Operand::PointerReg(2));
        op.is_vector = true;
        op.from_type = Some(ElementType::Int16);
        op.element_type = Some(ElementType::Int32);

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
        // Create a device with tiles. Tile(1,3) is the executing core.
        // Neighbor mapping (CardDir model):
        //   South = tile(1, 2) [CardDir 4]
        //   West  = tile(0, 3) [CardDir 5]
        //   North = tile(1, 4) [CardDir 6]
        let mut device = crate::device::DeviceState::new_npu1();

        // Write known data into south neighbor's memory
        if let Some(south) = device.tile_mut(1, 2) {
            south.data_memory_mut()[0x100..0x104].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
        }

        // Write known data into west neighbor's memory
        if let Some(west) = device.tile_mut(0, 3) {
            west.data_memory_mut()[0x200..0x204].copy_from_slice(&0xCAFE_BABEu32.to_le_bytes());
        }

        // Write known data into north neighbor's memory
        if let Some(north) = device.tile_mut(1, 4) {
            north.data_memory_mut()[0x300..0x304].copy_from_slice(&0x1234_5678u32.to_le_bytes());
        }

        // Build NeighborMemory for tile(1,3) and load snapshots
        let mut nbr = NeighborMemory::new(1, 3);
        nbr.ensure_snapshot(MemoryQuadrant::South, &device);
        nbr.ensure_snapshot(MemoryQuadrant::West, &device);
        nbr.ensure_snapshot(MemoryQuadrant::North, &device);

        // Verify reads from each direction
        let south_mem = nbr.get_memory(MemoryQuadrant::South).expect("South neighbor should exist");
        assert_eq!(
            u32::from_le_bytes([south_mem[0x100], south_mem[0x101], south_mem[0x102], south_mem[0x103]]),
            0xDEAD_BEEF
        );

        let west_mem = nbr.get_memory(MemoryQuadrant::West).expect("West neighbor should exist");
        assert_eq!(
            u32::from_le_bytes([west_mem[0x200], west_mem[0x201], west_mem[0x202], west_mem[0x203]]),
            0xCAFE_BABE
        );

        let north_mem = nbr.get_memory(MemoryQuadrant::North).expect("North neighbor should exist");
        assert_eq!(
            u32::from_le_bytes([north_mem[0x300], north_mem[0x301], north_mem[0x302], north_mem[0x303]]),
            0x1234_5678
        );
    }

    #[test]
    fn test_neighbor_memory_buffer_write_and_apply() {
        let mut device = crate::device::DeviceState::new_npu1();

        // Build NeighborMemory for tile(1,3)
        let mut nbr = NeighborMemory::new(1, 3);
        nbr.ensure_snapshot(MemoryQuadrant::West, &device);

        // Buffer a write to west neighbor at offset 0x400
        let write_data = 0xAAAA_BBBBu32.to_le_bytes();
        nbr.buffer_write(MemoryQuadrant::West, 0x400, &write_data);

        assert!(nbr.has_pending_writes());

        // Apply writes to the device
        nbr.apply_writes(&mut device);

        // Verify the write landed in the west neighbor tile (0,3)
        let west = device.tile(0, 3).unwrap();
        let mem = west.data_memory();
        let value = u32::from_le_bytes([mem[0x400], mem[0x401], mem[0x402], mem[0x403]]);
        assert_eq!(value, 0xAAAA_BBBB);
    }

    #[test]
    fn test_cross_tile_scalar_load() {
        // Test that a load with CardDir 5 address reads from the west
        // neighbor's memory snapshot.
        let mut device = crate::device::DeviceState::new_npu1();

        // Write test data to west neighbor (tile 0,3)
        if let Some(west) = device.tile_mut(0, 3) {
            west.write_data_u32(0x100, 0xFEED_FACE);
        }

        // Build neighbor snapshot for tile(1,3)
        let mut nbr = NeighborMemory::new(1, 3);
        nbr.ensure_snapshot(MemoryQuadrant::West, &device);

        // Read from CardDir 5 address: 0x50100 = west neighbor, offset 0x100
        let tile = Tile::compute(1, 3);
        let value = MemoryUnit::read_memory(&tile, 0x50100, MemWidth::Word, Some(&nbr));
        assert_eq!(value, 0xFEED_FACE);
    }

    #[test]
    fn test_cross_tile_vector_load() {
        let mut device = crate::device::DeviceState::new_npu1();

        // Write vector data to north neighbor (tile 1,4)
        let test_data = [10u32, 20, 30, 40, 50, 60, 70, 80];
        if let Some(north) = device.tile_mut(1, 4) {
            let mem = north.data_memory_mut();
            for (i, &val) in test_data.iter().enumerate() {
                let offset = 0x200 + i * 4;
                let bytes = val.to_le_bytes();
                mem[offset..offset + 4].copy_from_slice(&bytes);
            }
        }

        // Build neighbor snapshot
        let mut nbr = NeighborMemory::new(1, 3);
        nbr.ensure_snapshot(MemoryQuadrant::North, &device);

        // Read vector from CardDir 6: 0x60200 = north neighbor, offset 0x200
        let tile = Tile::compute(1, 3);
        let result = MemoryUnit::read_vector_from_memory(&tile, 0x60200, Some(&nbr));
        assert_eq!(result, test_data);
    }

    #[test]
    fn test_cross_tile_scalar_store_buffered() {
        // Cross-tile stores are buffered, not immediately written.
        let mut device = crate::device::DeviceState::new_npu1();

        let mut nbr = NeighborMemory::new(1, 3);
        nbr.ensure_snapshot(MemoryQuadrant::South, &device);

        // Write to CardDir 4 (south) address 0x40400
        let mut tile = Tile::compute(1, 3);
        MemoryUnit::write_memory(&mut tile, 0x40400, 0xBAAD_F00D, MemWidth::Word, Some(&mut nbr));

        // Should be buffered, not yet in the south neighbor
        let south = device.tile(1, 2).unwrap();
        let mem = south.data_memory();
        let value = u32::from_le_bytes([mem[0x400], mem[0x401], mem[0x402], mem[0x403]]);
        assert_eq!(value, 0, "Write should be buffered, not applied yet");

        // Apply writes
        assert!(nbr.has_pending_writes());
        nbr.apply_writes(&mut device);

        // Now it should be visible
        let south = device.tile(1, 2).unwrap();
        let mem = south.data_memory();
        let value = u32::from_le_bytes([mem[0x400], mem[0x401], mem[0x402], mem[0x403]]);
        assert_eq!(value, 0xBAAD_F00D);
    }

    #[test]
    fn test_cross_tile_vector_store_buffered() {
        let mut device = crate::device::DeviceState::new_npu1();

        let mut nbr = NeighborMemory::new(1, 3);
        nbr.ensure_snapshot(MemoryQuadrant::West, &device);

        // Write vector to CardDir 5 (west) address 0x50800
        let test_data = [1u32, 2, 3, 4, 5, 6, 7, 8];
        let mut tile = Tile::compute(1, 3);
        MemoryUnit::write_vector_to_memory(&mut tile, 0x50800, test_data, Some(&mut nbr));

        // Apply and verify
        nbr.apply_writes(&mut device);

        let west = device.tile(0, 3).unwrap();
        let mem = west.data_memory();
        for (i, &expected) in test_data.iter().enumerate() {
            let offset = 0x800 + i * 4;
            let actual = u32::from_le_bytes([mem[offset], mem[offset+1], mem[offset+2], mem[offset+3]]);
            assert_eq!(actual, expected, "Vector word {} mismatch", i);
        }
    }

    #[test]
    fn test_local_fast_path_with_neighbors() {
        // Verify that local memory (CardDir 7 = East) works correctly
        // even when NeighborMemory is Some.
        let device = crate::device::DeviceState::new_npu1();
        let mut nbr = NeighborMemory::new(1, 3);
        nbr.ensure_snapshot(MemoryQuadrant::West, &device);

        let mut tile = make_tile();
        tile.write_data_u32(0x100, 0x42424242);

        // Read from local memory (CardDir 7) with neighbors present
        let value = MemoryUnit::read_memory(&tile, 0x70100, MemWidth::Word, Some(&nbr));
        assert_eq!(value, 0x42424242);

        // Write to local memory (CardDir 7) with neighbors present
        MemoryUnit::write_memory(&mut tile, 0x70200, 0x99887766, MemWidth::Word, Some(&mut nbr));
        assert_eq!(tile.read_data_u32(0x200), Some(0x99887766));

        // No cross-tile writes should have been buffered
        assert!(!nbr.has_pending_writes());
    }

    #[test]
    fn test_edge_tile_no_west_neighbor() {
        // Tile(0,3) has no west neighbor (col 0).
        // Cross-tile read should return 0.
        let device = crate::device::DeviceState::new_npu1();
        let mut nbr = NeighborMemory::new(0, 3);
        nbr.ensure_snapshot(MemoryQuadrant::West, &device);

        let tile = Tile::compute(0, 3);
        // CardDir 5 = West, but col 0 has no west neighbor
        let value = MemoryUnit::read_memory(&tile, 0x50100, MemWidth::Word, Some(&nbr));
        assert_eq!(value, 0, "Read from non-existent west neighbor should return 0");
    }

    #[test]
    fn test_load_latency_local_vs_cross_tile() {
        use crate::interpreter::timing::CROSS_TILE_LATENCY;

        // Local memory (CardDir 7 = East = local): base latency only
        assert_eq!(load_latency_for_address(0x70000), LATENCY_MEMORY as u64);
        assert_eq!(load_latency_for_address(0x7FFFF), LATENCY_MEMORY as u64);

        // West neighbor (CardDir 5): base + cross-tile
        let expected_cross = LATENCY_MEMORY as u64 + CROSS_TILE_LATENCY as u64;
        assert_eq!(load_latency_for_address(0x50000), expected_cross);
        assert_eq!(load_latency_for_address(0x5FFFF), expected_cross);

        // North neighbor (CardDir 6): base + cross-tile
        assert_eq!(load_latency_for_address(0x60000), expected_cross);

        // South neighbor (CardDir 4): base + cross-tile
        assert_eq!(load_latency_for_address(0x40000), expected_cross);

        // CardDir 7 at different offset: still local
        assert_eq!(load_latency_for_address(0x70440), LATENCY_MEMORY as u64);
    }

    #[test]
    fn test_carddir7_is_local_memory() {
        // On AIE2, the linker places data at 0x70000+ because CardDir 7
        // (East) IS the local tile's data memory. Verify that 0x70440
        // correctly accesses local memory at offset 0x440.
        let mut tile = make_tile();
        tile.write_data_u32(0x440, 0xABCDABCD);

        // 0x70440 = CardDir 7 = Local, offset 0x0440
        let value = MemoryUnit::read_memory(&tile, 0x70440, MemWidth::Word, None);
        assert_eq!(value, 0xABCDABCD);
    }

    // === read_register_pure tests (direct, not through load/store) ===
    //
    // Note: core load/store instructions use the 20-bit CardDir-routed data
    // address space (CardDir = addr / MEMORY_SIZE, offset = addr & 0xFFFF).
    // Register access happens through dedicated paths (CDO, lock instructions,
    // control register writes), NOT through load/store MMIO.
    // read_register_pure exists for those dedicated paths.

    #[test]
    fn test_read_register_pure_dma_bd() {
        let mut tile = make_tile();
        tile.dma_bds[0].addr_low = 0xBEEF_CAFE;
        // Direct read_register_pure at DMA BD offset
        assert_eq!(tile.read_register_pure(0x1D000), 0xBEEF_CAFE);
    }

    #[test]
    fn test_read_register_pure_lock_value() {
        let mut tile = make_tile();
        tile.locks[3].set(7);
        let reg_layout = crate::device::regdb::device_reg_layout();
        let lock3_offset = reg_layout.memory_lock_base + 3 * reg_layout.memory_lock_stride;
        assert_eq!(tile.read_register_pure(lock3_offset), 7);
    }

    // =====================================================================
    // SP-Relative Addressing Tests
    // =====================================================================

    #[test]
    fn test_sp_relative_store_and_load() {
        // Test the fundamental save/restore pattern:
        //   st p7, [sp, #-32]    -- save p7 to stack
        //   lda p7, [sp, #-32]   -- restore p7 from stack
        //
        // This is the exact pattern used in function prologues/epilogues.
        use crate::interpreter::state::SP_PTR_INDEX;

        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Set up: sp = 0x70060, p7 = 0x78000
        ctx.set_sp(0x70060);
        ctx.pointer.write(7, 0x78000);

        // Store p7 to [sp - 32].
        // The store has: sources[0]=value(ScalarReg from p7), Memory{base=SP, offset=-32}
        // In the kernel layout: sources contain Memory operand.
        // For simplicity, construct as: dest=value, source=Memory{SP, -32}
        //
        // Actually the kernel store layout is:
        //   sources[0] = value (PointerReg(7) treated as ScalarReg for store)
        //   sources[1..] = Memory or Pointer for address
        //
        // But the test store layout uses dest=value, source=ptr.
        // Let me use the Memory operand directly.

        // First, compute the address manually and write directly
        // to verify the address computation is correct.
        let store_addr = ctx.sp().wrapping_add(-32_i32 as u32);
        let (quadrant, local_offset) = decode_data_address(store_addr);
        assert_eq!(quadrant, MemoryQuadrant::Local, "Stack should be in local memory");
        assert_eq!(local_offset, 0x0040, "0x70060 - 32 = 0x70040, CardDir 7 local, offset 0x40");

        // Write p7's value to memory at the stack slot
        let p7_val = ctx.pointer_read(7);
        assert_eq!(p7_val, 0x78000);
        MemoryUnit::write_memory(&mut tile, store_addr, p7_val as u64, MemWidth::Word, None);

        // Verify the value is in memory
        assert_eq!(tile.read_data_u32(local_offset), Some(0x78000));

        // Now clobber p7 (simulating mov p7, sp in the prologue)
        ctx.pointer.write(7, ctx.sp());
        assert_eq!(ctx.pointer.read(7), 0x70060);

        // Load p7 back from [sp - 32] using the Memory operand
        let load_op = SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load)
            .with_dest(Operand::PointerReg(7))
            .with_source(Operand::Memory { base: SP_PTR_INDEX, offset: -32 });

        MemoryUnit::execute(&load_op, &mut ctx, &mut tile, None);
        ctx.flush_pending_writes();

        assert_eq!(ctx.pointer.read(7), 0x78000,
            "p7 should be restored to original value after load from [sp, #-32]");
    }

    #[test]
    fn test_sp_relative_load_with_latency() {
        // Same as above but verify the load goes through the pending write
        // queue with proper latency (not flushed immediately).
        use crate::interpreter::state::SP_PTR_INDEX;
        use crate::interpreter::timing::LATENCY_MEMORY;

        let mut ctx = make_ctx();
        let mut tile = make_tile();

        ctx.set_sp(0x70080);
        ctx.cycles = 100;

        // Write test value to [sp - 32] = 0x70060 -> local 0x0060
        let addr = ctx.sp().wrapping_add(-32_i32 as u32);
        MemoryUnit::write_memory(&mut tile, addr, 0x78000, MemWidth::Word, None);

        // Load from [sp, #-32] into p7
        let load_op = SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load)
            .with_dest(Operand::PointerReg(7))
            .with_source(Operand::Memory { base: SP_PTR_INDEX, offset: -32 });

        MemoryUnit::execute(&load_op, &mut ctx, &mut tile, None);

        // Should NOT be committed yet (load latency = 7)
        assert_eq!(ctx.pointer.read(7), 0,
            "Load should not be committed immediately (latency pending)");

        // Before ready_cycle: read returns old value
        ctx.cycles = 101;
        assert_eq!(ctx.pointer_read(7), 0,
            "Before ready_cycle, should return old register value");

        // At ready_cycle: forward works, then commit
        ctx.cycles = 100 + LATENCY_MEMORY as u64;
        assert_eq!(ctx.pointer_read(7), 0x78000,
            "At ready_cycle, forward should return pending load value");
        ctx.commit_pending_writes();
        assert_eq!(ctx.pointer.read(7), 0x78000,
            "After commit, p7 should have restored value");
    }

    #[test]
    fn test_sp_relative_store_via_memory_operand() {
        // Test store using Memory operand with SP_PTR_INDEX as base.
        // st r5, [sp, #-8]
        use crate::interpreter::state::SP_PTR_INDEX;

        let mut ctx = make_ctx();
        let mut tile = make_tile();

        ctx.set_sp(0x70040);
        ctx.scalar.write(5, 0xDEAD_BEEF);

        // Store r5 to [sp - 8] using kernel layout:
        // sources[0] = value (ScalarReg(5))
        // sources[1] = Memory { base: SP_PTR_INDEX, offset: -8 }
        let store_op = SlotOp::from_semantic(SlotIndex::Store, SemanticOp::Store)
            .with_source(Operand::ScalarReg(5))
            .with_source(Operand::Memory { base: SP_PTR_INDEX, offset: -8 });

        MemoryUnit::execute(&store_op, &mut ctx, &mut tile, None);

        // Verify: [sp - 8] = 0x70040 - 8 = 0x70038 -> local 0x0038
        assert_eq!(tile.read_data_u32(0x38), Some(0xDEAD_BEEF),
            "Store via SP-relative addressing should write to correct local offset");
    }

    #[test]
    fn test_full_save_clobber_restore_with_forwarding() {
        // Full prologue/epilogue sequence with cycle-accurate timing:
        //   Cycle 10: st p7, [sp, #-32]       (save p7 = 0x78000)
        //   Cycle 11: mov p7, sp               (clobber p7 = sp)
        //   Cycle 12: commit p7 = sp           (latency 1)
        //   ... (function body, p7 used as frame pointer) ...
        //   Cycle 50: lda p7, [sp, #-32]       (restore, latency 7, ready=57)
        //   Cycles 51-56: reads return clobbered sp value (load not ready)
        //   Cycle 57: load ready, forwarding returns 0x78000
        use crate::interpreter::state::SP_PTR_INDEX;

        let mut ctx = make_ctx();
        let mut tile = make_tile();

        ctx.set_sp(0x70060);
        ctx.pointer.write(7, 0x78000);

        // Cycle 10: Store p7 to [sp - 32] (Memory-based address)
        ctx.cycles = 10;
        let store_addr = ctx.sp().wrapping_add(-32_i32 as u32);
        MemoryUnit::write_memory(&mut tile, store_addr, ctx.pointer_read(7) as u64, MemWidth::Word, None);

        // Cycle 11: mov p7, sp (clobber)
        ctx.cycles = 11;
        ctx.queue_pointer_write(7, ctx.sp(), 1); // ready=12

        // Verify p7 is still old (deferred by 1)
        assert_eq!(ctx.pointer.read(7), 0x78000,
            "p7 live should still be old value (write deferred)");
        // But forward returns new value
        ctx.cycles = 12;
        assert_eq!(ctx.pointer_read(7), 0x70060,
            "p7 forward should return sp value after clobber");

        // Commit the clobber
        ctx.commit_pending_writes();
        assert_eq!(ctx.pointer.read(7), 0x70060, "p7 committed to sp value");

        // ... function body ...

        // Cycle 50: Restore p7 from [sp, #-32]
        ctx.cycles = 50;
        let load_op = SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load)
            .with_dest(Operand::PointerReg(7))
            .with_source(Operand::Memory { base: SP_PTR_INDEX, offset: -32 });
        MemoryUnit::execute(&load_op, &mut ctx, &mut tile, None);

        // Cycle 51: load not ready -- read returns clobbered sp value
        ctx.cycles = 51;
        assert_eq!(ctx.pointer_read(7), 0x70060,
            "Before ready_cycle, should read clobbered sp value");

        // Cycle 55: still not ready
        ctx.cycles = 55;
        assert_eq!(ctx.pointer_read(7), 0x70060,
            "Still before ready_cycle, should read clobbered sp value");

        // Cycle 57: load ready -- forward returns restored value, then commit
        ctx.cycles = 57;
        assert_eq!(ctx.pointer_read(7), 0x78000,
            "At ready_cycle, forwarding should return restored p7 value");
        ctx.commit_pending_writes();
        assert_eq!(ctx.pointer.read(7), 0x78000,
            "p7 should be committed to restored value");
    }

    #[test]
    fn test_post_modify_with_sp_base() {
        // Verify that apply_post_modify correctly handles SP_PTR_INDEX.
        // If post-modify occurs on an SP-relative load, it should modify SP,
        // not alias to p7.
        use crate::interpreter::state::SP_PTR_INDEX;

        let mut ctx = make_ctx();
        let mut tile = make_tile();

        ctx.set_sp(0x70100);
        ctx.pointer.write(7, 0xBEEF); // p7 should NOT be modified
        tile.write_data_u32(0x100, 0x42);

        // lda r0, [sp], #4 -- load from sp, then sp += 4
        let load_op = SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load)
            .with_post_modify(PostModify::Immediate(4))
            .with_dest(Operand::ScalarReg(0))
            .with_source(Operand::Memory { base: SP_PTR_INDEX, offset: 0 });

        MemoryUnit::execute(&load_op, &mut ctx, &mut tile, None);

        // SP should be post-modified: 0x70100 + 4 = 0x70104
        assert_eq!(ctx.sp(), 0x70104,
            "SP should be post-modified by +4");

        // p7 should be UNAFFECTED
        assert_eq!(ctx.pointer.read(7), 0xBEEF,
            "p7 must not be affected by SP post-modify");

        ctx.flush_pending_writes();
        assert_eq!(ctx.scalar.read(0), 0x42);
    }

    // =====================================================================
    // Pointer Register Store/Load (callee-save pattern)
    // =====================================================================

    /// Build a store SlotOp that mimics TableGen-decoded output.
    /// Sets encoding_name so get_store_value uses the LLVM convention
    /// (sources[0] = value) instead of the legacy heuristic.
    fn make_decoded_store() -> SlotOp {
        let mut op = SlotOp::from_semantic(SlotIndex::Store, SemanticOp::Store);
        op.encoding_name = Some("ST_dms_spill".into());
        op
    }

    #[test]
    fn test_store_pointer_register_value() {
        // `st p7, [sp, #-32]` -- stores p7's VALUE to memory.
        // LLVM's InOperandList puts the value at sources[0] for all stores.
        // The mSclSt register class covers ScalarReg, PointerReg, ModifierReg,
        // and LR, so the value can be any of these types.
        let mut ctx = make_ctx();
        let mut tile = make_tile();
        ctx.set_sp(0x100);
        ctx.queue_pointer_write(7, 0x78000, 1);
        ctx.cycles = 1;
        ctx.commit_pending_writes();
        assert_eq!(ctx.pointer.read(7), 0x78000);

        // st p7, [sp, #-32]: sources[0]=PointerReg(7), sources[1]=Memory{sp, -32}
        let op = make_decoded_store()
            .with_source(Operand::PointerReg(7))
            .with_source(Operand::Memory { base: SP_PTR_INDEX, offset: -32 });

        MemoryUnit::execute(&op, &mut ctx, &mut tile, None);

        // Memory at sp-32 = 0x100-0x20 = 0xE0 should contain 0x78000
        let stored = tile.read_data_u32(0xE0);
        assert_eq!(stored, Some(0x78000),
            "st p7 must store p7's VALUE (0x78000), not 0");
    }

    #[test]
    fn test_store_and_load_pointer_register_roundtrip() {
        // Full callee-save/restore: st p7 -> clobber p7 -> lda p7
        let mut ctx = make_ctx();
        let mut tile = make_tile();
        ctx.set_sp(0x100);
        ctx.queue_pointer_write(7, 0x78000, 1);
        ctx.cycles = 1;
        ctx.commit_pending_writes();

        // Store p7 to stack (TableGen path)
        let store_op = make_decoded_store()
            .with_source(Operand::PointerReg(7))
            .with_source(Operand::Memory { base: SP_PTR_INDEX, offset: -32 });
        MemoryUnit::execute(&store_op, &mut ctx, &mut tile, None);

        // Clobber p7
        ctx.cycles = 2;
        ctx.queue_pointer_write(7, 0xDEAD, 1);
        ctx.cycles = 3;
        ctx.commit_pending_writes();
        assert_eq!(ctx.pointer.read(7), 0xDEAD);

        // Load p7 back from stack
        ctx.cycles = 10;
        let load_op = SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load)
            .with_dest(Operand::PointerReg(7))
            .with_source(Operand::Memory { base: SP_PTR_INDEX, offset: -32 });
        MemoryUnit::execute(&load_op, &mut ctx, &mut tile, None);
        ctx.flush_pending_writes();

        assert_eq!(ctx.pointer.read(7), 0x78000,
            "p7 must be restored to original value after store/load roundtrip");
    }

    #[test]
    fn test_store_pointer_register_p6() {
        // Verify the fix works for all pointer registers, not just p7
        let mut ctx = make_ctx();
        let mut tile = make_tile();
        ctx.set_sp(0x200);
        ctx.queue_pointer_write(6, 0x74000, 1);
        ctx.cycles = 1;
        ctx.commit_pending_writes();

        let op = make_decoded_store()
            .with_source(Operand::PointerReg(6))
            .with_source(Operand::Memory { base: SP_PTR_INDEX, offset: -36 });

        MemoryUnit::execute(&op, &mut ctx, &mut tile, None);

        // sp-36 = 0x200-0x24 = 0x1DC
        let stored = tile.read_data_u32(0x1DC);
        assert_eq!(stored, Some(0x74000),
            "st p6 must store p6's value correctly");
    }

    #[test]
    fn test_store_modifier_register() {
        // mSclSt also covers modifier registers (eDJ, eM).
        // Verify st djN, [sp, #off] works via the TableGen path.
        let mut ctx = make_ctx();
        let mut tile = make_tile();
        ctx.set_sp(0x100);
        ctx.modifier.write(0, 0x42);

        let op = make_decoded_store()
            .with_source(Operand::ModifierReg(0))
            .with_source(Operand::Memory { base: SP_PTR_INDEX, offset: -4 });

        MemoryUnit::execute(&op, &mut ctx, &mut tile, None);

        let stored = tile.read_data_u32(0xFC); // 0x100 - 4
        assert_eq!(stored, Some(0x42),
            "st dj0 must store modifier register value");
    }
}
