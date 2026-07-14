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

mod neighbor;
pub use neighbor::NeighborMemory;

use crate::device::state::NeighborView;

use crate::device::tile::Tile;
use crate::interpreter::bundle::{ElementType, MemWidth, Operand, PostModify, SlotIndex, SlotOp};
use crate::interpreter::state::ExecutionContext;
use crate::interpreter::timing::{LATENCY_MEMORY, MemoryQuadrant};
use xdna_archspec::aie2::isa::SemanticOp;
use xdna_archspec::aie2::memory_map;

/// Offset mask for extracting the local address within a tile's memory.
/// `address & OFFSET_MASK` gives the byte offset within the target tile.
const OFFSET_MASK: u32 = xdna_archspec::aie2::compute::MEMORY_SIZE as u32 - 1;

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
///
/// Exposed `pub(crate)` for the same reason as [`MemoryUnit::get_address`]:
/// `CycleAccurateExecutor::peek_bank_demand` needs the same Local-vs-neighbour
/// scoping the load/store sites use, to exclude cross-tile accesses from the
/// bank-demand mask without re-deriving the quadrant decode.
#[inline]
pub(crate) fn decode_data_address(addr: u32) -> (MemoryQuadrant, usize) {
    let offset = (addr & OFFSET_MASK) as usize;
    (MemoryQuadrant::from_address(addr), offset)
}

/// Compute the load latency for a core data memory access.
///
/// On AIE2, all data memory accesses (local and cross-tile neighbor) have the
/// same pipeline depth (7 cycles). The core accesses neighbor memory modules
/// through direct ports, NOT through the stream switch. Cross-tile routing
/// latency applies only to DMA transfers, not core load/store operations.
///
/// The Chess compiler schedules load-use distances based on this uniform
/// latency; adding extra cycles for cross-tile accesses causes pipeline
/// hazards where stores read stale register values.
#[inline]
fn load_latency_for_address(_addr: u32) -> u64 {
    LATENCY_MEMORY as u64
}

// NeighborMemory is in neighbor.rs

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
        view: Option<&NeighborView>,
    ) -> bool {
        // PostModify comes from op.post_modify (populated from the AG field
        // during decode).
        let pm = &op.post_modify;

        match op.semantic {
            Some(SemanticOp::Load) if !op.is_vector => {
                // Core-relay recorder (P6): capture local-buffer loads before delegating.
                // addr >> 16 == 7 selects the local data-memory quadrant (CardDir 7 = East =
                // local on AIE2). Stack and cross-tile accesses are naturally excluded.
                let addr = Self::get_address(op, ctx);
                if addr >> 16 == 7 {
                    if let Some((_, buf_evs)) = &mut tile.core_relay_recorder {
                        buf_evs.push(crate::device::tile::CoreBufEvent {
                            cycle: ctx.cycles,
                            local_off: addr & 0xFFFF,
                            is_store: false,
                            col: tile.col,
                            row: tile.row,
                        });
                    }
                }
                Self::execute_load(op, ctx, tile, op.mem_width, pm, neighbors, view);
                true
            }

            Some(SemanticOp::Store) if !op.is_vector => {
                // Core-relay recorder (P6): capture local-buffer stores.
                let addr = Self::get_store_address(op, ctx);
                if addr >> 16 == 7 {
                    if let Some((_, buf_evs)) = &mut tile.core_relay_recorder {
                        buf_evs.push(crate::device::tile::CoreBufEvent {
                            cycle: ctx.cycles,
                            local_off: addr & 0xFFFF,
                            is_store: true,
                            col: tile.col,
                            row: tile.row,
                        });
                    }
                }
                Self::execute_store(op, ctx, tile, op.mem_width, pm, neighbors, view);
                true
            }

            Some(SemanticOp::Load) if op.slot == SlotIndex::LoadA => {
                // Core-relay recorder (P6): capture vector-A loads to local buffer space.
                let addr = Self::get_address(op, ctx);
                if addr >> 16 == 7 {
                    if let Some((_, buf_evs)) = &mut tile.core_relay_recorder {
                        buf_evs.push(crate::device::tile::CoreBufEvent {
                            cycle: ctx.cycles,
                            local_off: addr & 0xFFFF,
                            is_store: false,
                            col: tile.col,
                            row: tile.row,
                        });
                    }
                }
                Self::execute_vector_load_a(op, ctx, tile, pm, neighbors, view);
                true
            }

            Some(SemanticOp::Load) if op.slot == SlotIndex::LoadB => {
                // VLDB_4x: 4-way gather load using vector elements as addresses.
                // Source register contains 4 x 32-bit memory addresses in its
                // lo or hi half. Hardware reads 64 bits from each address.
                // Core-relay recorder (P6): capture vector-B loads to local buffer space.
                let addr = Self::get_address(op, ctx);
                if addr >> 16 == 7 {
                    if let Some((_, buf_evs)) = &mut tile.core_relay_recorder {
                        buf_evs.push(crate::device::tile::CoreBufEvent {
                            cycle: ctx.cycles,
                            local_off: addr & 0xFFFF,
                            is_store: false,
                            col: tile.col,
                            row: tile.row,
                        });
                    }
                }
                if Self::is_load_4x(op) {
                    Self::execute_vector_load_4x(op, ctx, tile, neighbors, view);
                } else {
                    Self::execute_vector_load_b(op, ctx, tile, pm, neighbors, view);
                }
                true
            }

            Some(SemanticOp::Unpack) => {
                let from_type = op.from_type.unwrap_or(ElementType::Int32);
                let to_type = op.element_type.unwrap_or(ElementType::Int32);
                Self::execute_vector_load_unpack(op, ctx, tile, from_type, to_type, pm);
                true
            }

            Some(SemanticOp::Store) if op.is_vector => {
                // Core-relay recorder (P6): capture vector stores to local buffer space.
                let addr = Self::get_store_address(op, ctx);
                if addr >> 16 == 7 {
                    if let Some((_, buf_evs)) = &mut tile.core_relay_recorder {
                        buf_evs.push(crate::device::tile::CoreBufEvent {
                            cycle: ctx.cycles,
                            local_off: addr & 0xFFFF,
                            is_store: true,
                            col: tile.col,
                            row: tile.row,
                        });
                    }
                }
                Self::execute_vector_store(op, ctx, tile, pm, neighbors, view);
                true
            }

            // ========== Fused load+compute operations ==========
            Some(SemanticOp::Ups) if Self::has_memory_operand(op) => {
                // vlda.ups: load from memory, upshift to accumulator
                Self::execute_fused_load_ups(op, ctx, tile, pm);
                true
            }

            Some(SemanticOp::Convert) if Self::is_load_slot(op) && Self::has_memory_operand(op) => {
                // vlda.conv: load from memory, convert (e.g., bf16 -> f32)
                Self::execute_fused_load_convert(op, ctx, tile, pm);
                true
            }

            // ========== Fused compute+store operations ==========
            //
            // Only handle these if the instruction has a memory address operand.
            // Standalone SRS/Pack/Convert (e.g., VSRS in the ST slot) have no
            // memory operand -- they write to a register, not memory. VectorAlu
            // handles those.
            Some(SemanticOp::Srs) if Self::has_memory_operand(op) => {
                // vst.srs: shift-round-saturate accumulator, store to memory
                Self::execute_fused_store_srs(op, ctx, tile, pm, neighbors, view);
                true
            }

            Some(SemanticOp::Pack) if Self::has_memory_operand(op) => {
                // vst.pack: pack vector, store narrowed data to memory
                Self::execute_fused_store_pack(op, ctx, tile, pm, neighbors, view);
                true
            }

            Some(SemanticOp::Convert) if Self::has_memory_operand(op) => {
                // vst.conv: convert (e.g., f32 -> bf16), store to memory
                Self::execute_fused_store_convert(op, ctx, tile, pm, neighbors, view);
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
        view: Option<&NeighborView>,
    ) {
        // Get address from source operand
        let addr = Self::get_address(op, ctx);
        let latency = load_latency_for_address(addr);

        // Track bank access for conflict detection (local memory only)
        let (quadrant, local_offset) = decode_data_address(addr);
        if quadrant == MemoryQuadrant::Local {
            ctx.record_core_bank_access(local_offset as u32, width.bytes() as usize, tile.bank_layout());
        }

        log::trace!(
            "[LOAD] addr=0x{:X} width={:?} dest={:?} srcs={:?} latency={}",
            addr,
            width,
            op.dest,
            op.sources,
            latency
        );
        // Handle vector loads specially
        if width == MemWidth::Vector256 {
            let vec_data = Self::read_vector_from_memory(tile, addr, neighbors, view);
            if crate::debug::watch::is_watched(addr as u64, 32) {
                let dest_str = op.dest.as_ref().map(|d| format!("{:?}", d)).unwrap_or_default();
                crate::debug::watch::log_core_load(
                    ctx.cycles,
                    tile.col,
                    tile.row,
                    ctx.pc(),
                    addr as u64,
                    vec_data[0],
                    &dest_str,
                );
            }
            if let Some(dest) = &op.dest {
                ctx.queue_vector_load(dest.clone(), vec_data, latency);
            }
        } else if width == MemWidth::QuadWord {
            // 128-bit load: read 16 bytes into a q register (mask) or vector.
            let full_data = Self::read_vector_from_memory(tile, addr, neighbors, view);
            let mut vec_data = [0u32; 8];
            vec_data[0] = full_data[0];
            vec_data[1] = full_data[1];
            vec_data[2] = full_data[2];
            vec_data[3] = full_data[3];
            // words 4-7 stay zero
            if crate::debug::watch::is_watched(addr as u64, 16) {
                let dest_str = op.dest.as_ref().map(|d| format!("{:?}", d)).unwrap_or_default();
                crate::debug::watch::log_core_load(
                    ctx.cycles,
                    tile.col,
                    tile.row,
                    ctx.pc(),
                    addr as u64,
                    vec_data[0],
                    &dest_str,
                );
            }
            if let Some(dest) = &op.dest {
                ctx.queue_vector_load(dest.clone(), vec_data, latency);
            }
        } else {
            // Scalar/partial loads go through write_dest_with_latency (which queues)
            let raw_value = Self::read_memory(tile, addr, width, neighbors, view);

            // Apply sign/zero extension based on element type.
            // lda.s8 sign-extends 8-bit to 32-bit, lda.u8 zero-extends, etc.
            // read_memory returns zero-extended by default, so we only need
            // to fix up signed narrow loads.
            let value = if let Some(ref et) = op.element_type {
                if et.is_signed() {
                    match width {
                        MemWidth::Byte => (raw_value as u8 as i8 as i32 as u32) as u64,
                        MemWidth::HalfWord => (raw_value as u16 as i16 as i32 as u32) as u64,
                        _ => raw_value,
                    }
                } else {
                    raw_value // already zero-extended
                }
            } else {
                raw_value // no element type info, keep as-is
            };

            if crate::debug::watch::is_watched(addr as u64, width.bytes() as usize) {
                let dest_str = op.dest.as_ref().map(|d| format!("{:?}", d)).unwrap_or_default();
                crate::debug::watch::log_core_load(
                    ctx.cycles,
                    tile.col,
                    tile.row,
                    ctx.pc(),
                    addr as u64,
                    value as u32,
                    &dest_str,
                );
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
        view: Option<&NeighborView>,
    ) {
        // Get address using store-specific layout: sources[1]=ptr, sources[2]=offset
        let addr = Self::get_store_address(op, ctx);

        // Track bank access for conflict detection (local memory only)
        let (quadrant, local_offset) = decode_data_address(addr);
        if quadrant == MemoryQuadrant::Local {
            ctx.record_core_bank_access(local_offset as u32, width.bytes() as usize, tile.bank_layout());
        }

        // Handle vector/wide stores
        if width == MemWidth::Vector256 || width == MemWidth::QuadWord {
            // Read data from VectorReg, AccumReg, or ControlReg source.
            let vec_data = Self::read_store_data_wide(op, ctx);

            if let Some(ref data) = vec_data {
                if crate::debug::watch::is_watched(
                    addr as u64,
                    if width == MemWidth::QuadWord { 16 } else { 32 },
                ) {
                    crate::debug::watch::log_core_store(
                        ctx.cycles,
                        tile.col,
                        tile.row,
                        ctx.pc(),
                        addr as u64,
                        data[0],
                    );
                }
                if width == MemWidth::QuadWord {
                    // 128-bit store: write only lower 4 words (16 bytes).
                    let half = [data[0], data[1], data[2], data[3], 0, 0, 0, 0];
                    Self::write_vector_to_memory(tile, addr, half, neighbors, view);
                } else {
                    Self::write_vector_to_memory(tile, addr, *data, neighbors, view);
                }
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
                if crate::debug::watch::is_watched(addr as u64, width.bytes() as usize) {
                    crate::debug::watch::log_core_store(
                        ctx.cycles,
                        tile.col,
                        tile.row,
                        ctx.pc(),
                        addr as u64,
                        value as u32,
                    );
                }
                Self::write_memory(tile, addr, value, width, neighbors, view);
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
        view: Option<&NeighborView>,
    ) {
        let addr = Self::get_address(op, ctx);
        log::trace!("[VLDA] addr=0x{:X} dest={:?}", addr, op.dest);
        let latency = load_latency_for_address(addr);

        // Track bank access for conflict detection (local memory only)
        let (quadrant, local_offset) = decode_data_address(addr);
        if quadrant == MemoryQuadrant::Local {
            ctx.record_core_bank_access(local_offset as u32, 32, tile.bank_layout());
        }

        // Read from memory -- respect 128-bit vs 256-bit width.
        let vec_data = if op.mem_width == MemWidth::QuadWord {
            // 128-bit load: only lower 4 words, upper 4 zeroed.
            let full = Self::read_vector_from_memory(tile, addr, neighbors, view);
            [full[0], full[1], full[2], full[3], 0, 0, 0, 0]
        } else {
            Self::read_vector_from_memory(tile, addr, neighbors, view)
        };

        if crate::debug::watch::is_watched(
            addr as u64,
            if op.mem_width == MemWidth::QuadWord {
                16
            } else {
                32
            },
        ) {
            let dest_str = op.dest.as_ref().map(|d| format!("{:?}", d)).unwrap_or_default();
            crate::debug::watch::log_core_load(
                ctx.cycles,
                tile.col,
                tile.row,
                ctx.pc(),
                addr as u64,
                vec_data[0],
                &dest_str,
            );
        }

        // Queue deferred write to destination register.
        // AccumReg destinations (AM loads) need width metadata for quarter writes.
        if let Some(dest) = &op.dest {
            match dest {
                Operand::AccumReg(_) => {
                    let width = op
                        .accum_width
                        .unwrap_or(crate::interpreter::decode::register_map::AccumWidth::Half);
                    ctx.queue_accum_load(dest.clone(), vec_data, width, latency);
                }
                _ => {
                    ctx.queue_vector_load(dest.clone(), vec_data, latency);
                }
            }
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
        view: Option<&NeighborView>,
    ) {
        let addr = Self::get_address(op, ctx);
        let latency = load_latency_for_address(addr);

        // Track bank access for conflict detection (local memory only)
        let (quadrant, local_offset) = decode_data_address(addr);
        if quadrant == MemoryQuadrant::Local {
            ctx.record_core_bank_access(local_offset as u32, 32, tile.bank_layout());
        }

        // Read from memory -- respect 128-bit vs 256-bit width.
        let vec_data = if op.mem_width == MemWidth::QuadWord {
            let full = Self::read_vector_from_memory(tile, addr, neighbors, view);
            [full[0], full[1], full[2], full[3], 0, 0, 0, 0]
        } else {
            Self::read_vector_from_memory(tile, addr, neighbors, view)
        };

        if crate::debug::watch::is_watched(
            addr as u64,
            if op.mem_width == MemWidth::QuadWord {
                16
            } else {
                32
            },
        ) {
            let dest_str = op.dest.as_ref().map(|d| format!("{:?}", d)).unwrap_or_default();
            crate::debug::watch::log_core_load(
                ctx.cycles,
                tile.col,
                tile.row,
                ctx.pc(),
                addr as u64,
                vec_data[0],
                &dest_str,
            );
        }

        // Queue deferred write to destination vector register
        if let Some(dest) = &op.dest {
            ctx.queue_vector_load(dest.clone(), vec_data, latency);
        }

        // Apply post-modify immediately (pointer update is not deferred)
        Self::apply_post_modify(op, ctx, post_modify);
    }

    /// Check if this is a VLDB_4x instruction (4-way gather load).
    fn is_load_4x(op: &SlotOp) -> bool {
        if let Some(name) = &op.encoding_name {
            let lower = name.to_ascii_lowercase();
            lower.contains("vldb.4x") || lower.contains("vldb_4x")
        } else {
            // Structural fallback: VLDB_4x has no pointer operand.
            // Regular VLDB has a PointerReg source; VLDB_4x has only VectorReg.
            op.sources.iter().all(|s| matches!(s, Operand::VectorReg(_))) && !op.sources.is_empty()
        }
    }

    /// Execute 4-way LUT load (VLDB_4x16/32/64 LO/HI).
    ///
    /// These instructions perform 4-way parallel memory reads for LUT
    /// (look-up table) access. The source vector provides 4 pre-computed
    /// addresses in its lo or hi half. The hardware applies alignment
    /// masking per element size and bank-interleaves odd-indexed reads.
    ///
    /// Semantics (from aietools me_addr.h):
    ///   1. Extract 4 addresses from the source register's lo/hi half
    ///   2. Mask each for alignment: 4x16=0xFFFFFFFC, 4x32=0xFFFFFFF8,
    ///      4x64=0xFFFFFFF0
    ///   3. OR 0x10 onto odd-indexed addresses (indices 1 and 3) for
    ///      memory bank interleaving
    ///   4. Read 64 bits from each of the 4 effective addresses
    ///   5. Pack results into the 256-bit destination register
    ///
    /// The bank interleave (`|0x10`) ensures parallel access to different
    /// 16-byte memory banks, enabling conflict-free 4-way reads when the
    /// base pointers (lut1/lut2) are 256-bit aligned.
    ///
    /// Source: aietools me_addr.h `lut_pointer()` + `load_4x_address()`.
    /// Confirmed by hardware characterization on Phoenix NPU (3/3 tests).
    fn execute_vector_load_4x(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        tile: &Tile,
        mut neighbors: Option<&mut NeighborMemory>,
        view: Option<&NeighborView>,
    ) {
        // Read source vector register.
        let src_reg = match op.sources.first() {
            Some(Operand::VectorReg(r)) => *r,
            _ => {
                log::warn!("[VLDB_4x] no VectorReg source");
                return;
            }
        };

        // Commit pending writes so a preceding vlda's data is visible.
        ctx.force_commit_all_pending();

        let src_data = ctx.vector.read(src_reg);

        // Parse encoding name for hi/lo and element size.
        let name = op.encoding_name.as_deref().unwrap_or("");
        let lower = name.to_ascii_lowercase();
        let is_hi = lower.contains("hi");

        let align_mask: u32 = if lower.contains("4x64") {
            0xFFFF_FFF0
        } else if lower.contains("4x32") {
            0xFFFF_FFF8
        } else {
            0xFFFF_FFFC // 4x16
        };

        // Extract 4 addresses from lo or hi half of source register.
        let raw_addrs = if is_hi {
            [src_data[4], src_data[5], src_data[6], src_data[7]]
        } else {
            [src_data[0], src_data[1], src_data[2], src_data[3]]
        };

        // Apply alignment mask, then bank-interleave odd indices.
        let eff_addrs = [
            raw_addrs[0] & align_mask,
            (raw_addrs[1] & align_mask) | 0x10,
            raw_addrs[2] & align_mask,
            (raw_addrs[3] & align_mask) | 0x10,
        ];

        // Read 64 bits from each effective address.
        let latency = LATENCY_MEMORY as u64;
        let mut result = [0u32; 8];
        for i in 0..4 {
            let addr = eff_addrs[i];
            let (quadrant, offset) = decode_data_address(addr);
            let mem: &[u8] = if quadrant == MemoryQuadrant::Local || neighbors.is_none() {
                tile.data_memory()
            } else {
                // Lazy snapshot per accessed quadrant. The mut re-borrow
                // ends at the if-let body so the shared `as_deref` below
                // can re-borrow `neighbors` for `get_memory`.
                if let (Some(n), Some(v)) = (neighbors.as_deref_mut(), view) {
                    n.ensure_snapshot(quadrant, v);
                }
                match neighbors.as_deref().and_then(|n| n.get_memory(quadrant)) {
                    Some(m) => m,
                    None => continue, // non-existent neighbor -> zeros
                }
            };

            if offset + 7 < mem.len() {
                result[i * 2] =
                    u32::from_le_bytes([mem[offset], mem[offset + 1], mem[offset + 2], mem[offset + 3]]);
                result[i * 2 + 1] =
                    u32::from_le_bytes([mem[offset + 4], mem[offset + 5], mem[offset + 6], mem[offset + 7]]);
            }
        }

        log::trace!(
            "[VLDB_4x] {} src=v{} addrs=[{:#x},{:#x},{:#x},{:#x}] -> [{:08x},{:08x},...]",
            name,
            src_reg,
            eff_addrs[0],
            eff_addrs[1],
            eff_addrs[2],
            eff_addrs[3],
            result[0],
            result[1],
        );

        if let Some(dest) = &op.dest {
            ctx.queue_vector_load(dest.clone(), result, latency);
        }
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
                    let vec_data = Self::read_vector_from_memory(tile, addr, None, None);
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

    // ========== Fused instruction helpers ==========

    /// Check if the op has a memory address operand (Memory { base, offset }).
    /// Fused instructions have this; standalone SRS/Pack/UPS on memory slots don't.
    fn has_memory_operand(op: &SlotOp) -> bool {
        // Shared with the VectorAlu skip gate: covers both offset (Memory{}) and
        // register-indirect (PointerReg: post-increment / indexed) addressing.
        op.addresses_memory()
    }

    /// Check if the op is on a load slot (LoadA or LoadB).
    fn is_load_slot(op: &SlotOp) -> bool {
        matches!(op.slot, SlotIndex::LoadA | SlotIndex::LoadB)
    }

    /// Load 256 bits from memory at the op's address. Shared by fused load handlers.
    fn fused_load_vector(op: &SlotOp, ctx: &mut ExecutionContext, tile: &Tile) -> [u32; 8] {
        let addr = Self::get_address(op, ctx);
        let (quadrant, local_offset) = decode_data_address(addr);
        if quadrant == MemoryQuadrant::Local {
            ctx.record_core_bank_access(local_offset as u32, 32, tile.bank_layout());
        }
        Self::read_vector_from_memory(tile, addr, None, None)
    }

    /// Get shift amount from operands (immediate or scalar register).
    fn get_shift_amount(op: &SlotOp, ctx: &ExecutionContext) -> u32 {
        for src in &op.sources {
            match src {
                Operand::Immediate(imm) => return *imm as u32,
                // The shift operand reads at pipeline stage E7, so a same-bundle
                // `MOV sN,#imm` (written at E1) forwards -- consult shift_forward
                // before the read-old snapshot. See ExecutionContext::shift_forward.
                Operand::ScalarReg(r) => {
                    return ctx.shift_forward(*r).unwrap_or_else(|| ctx.scalar_read(*r));
                }
                _ => {}
            }
        }
        0
    }

    /// Get accumulator register index from operands.
    fn get_acc_source(op: &SlotOp) -> u8 {
        for src in &op.sources {
            if let Operand::AccumReg(r) = src {
                return *r;
            }
        }
        0
    }

    /// Get vector register data from the first VectorReg source.
    fn get_vector_source(op: &SlotOp, ctx: &ExecutionContext) -> [u32; 8] {
        for src in &op.sources {
            if let Operand::VectorReg(r) = src {
                return ctx.vector.read(*r);
            }
        }
        [0u32; 8]
    }

    /// Find the first AccumReg index in the sources.
    fn find_acc_source(op: &SlotOp) -> Option<u8> {
        for src in &op.sources {
            if let Operand::AccumReg(r) = src {
                return Some(*r);
            }
        }
        None
    }

    /// Find the S-register shift operand, mirroring `get_shift_amount`'s
    /// operand precedence (an Immediate shift means no register sample).
    fn get_shift_scalar_reg(op: &SlotOp) -> Option<u8> {
        for src in &op.sources {
            match src {
                Operand::Immediate(_) => return None,
                Operand::ScalarReg(r) => return Some(*r),
                _ => {}
            }
        }
        None
    }

    // ========== Fused load+compute handlers ==========

    /// Execute `vlda.ups`: load from memory, upshift to accumulator.
    ///
    /// Loads a 256-bit vector from memory, then applies UPS (upshift) to
    /// widen narrow lanes into accumulator format. The result is written
    /// to the destination accumulator register.
    fn execute_fused_load_ups(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        tile: &Tile,
        post_modify: &PostModify,
    ) {
        let vec_data = Self::fused_load_vector(op, ctx, tile);
        let from = op.from_type.unwrap_or(ElementType::Int16);
        let to = op.element_type.unwrap_or(ElementType::Int32);

        // Half (bml/bmh) -> 512-bit single register.
        // Full/None (cm) -> 1024-bit wide pair via w2c upshift.
        let is_half =
            matches!(op.accum_width, Some(crate::interpreter::decode::register_map::AccumWidth::Half));

        match &op.dest {
            Some(Operand::AccumReg(r)) => {
                // II_VLDA_UPS reads its shift S-register at operand cycle 7
                // (issue+6), not at issue: the UPS conversion happens after
                // the load data returns, and the compiler schedules the
                // shift-setup `mov sN` AFTER the vlda.ups it controls
                // (verified against NPU1 silicon, vector fuzzer seeds
                // 1142-1448). Defer the conversion so the shift is sampled
                // at the stage-7 bundle. Immediate shifts have no register
                // to sample and convert at issue.
                if let Some(s) = Self::get_shift_scalar_reg(op) {
                    ctx.queue_pending_ups_load(vec_data, s, *r, from, to, is_half);
                } else {
                    let shift = Self::get_shift_amount(op, ctx);
                    if is_half {
                        let acc = super::vector_ups::ups_vector_to_acc(&vec_data, shift, from, to);
                        ctx.accumulator.write(*r, acc);
                    } else {
                        let acc_wide = super::vector_ups::ups_vector_to_acc_wide(&vec_data, shift, from, to);
                        ctx.accumulator.write_wide(*r, acc_wide);
                    }
                }
            }
            other => {
                log::warn!(
                    "[FUSED] vlda.ups: expected AccumReg dest, got {:?} (encoding={:?})",
                    other,
                    op.encoding_name,
                );
            }
        }

        Self::apply_post_modify(op, ctx, post_modify);
    }

    /// Execute `vlda.conv`: load from memory, convert type (e.g., bf16 -> f32).
    ///
    /// Loads a 256-bit vector from memory, then applies type conversion.
    /// For bf16->fp32, the 16 bf16 values expand to 16 fp32 values (512 bits)
    /// and are written to an accumulator register as float accumulator data.
    fn execute_fused_load_convert(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        tile: &Tile,
        post_modify: &PostModify,
    ) {
        let vec_data = Self::fused_load_vector(op, ctx, tile);
        let from = op.from_type.unwrap_or(ElementType::BFloat16);
        let to = op.element_type.unwrap_or(ElementType::Float32);

        match &op.dest {
            Some(Operand::AccumReg(r)) => {
                // bf16 -> fp32: 16 bf16 in 256 bits -> 16 fp32 in 512 bits.
                // Each bf16 is zero-extended to fp32 (left-shift by 16).
                // Store as 8 u64 words (2 x fp32 per u64).
                if from == ElementType::BFloat16 && to == ElementType::Float32 {
                    let mut acc = [0u64; 8];
                    for i in 0..8 {
                        let lo_bf16 = (vec_data[i] & 0xFFFF) as u64;
                        let hi_bf16 = ((vec_data[i] >> 16) & 0xFFFF) as u64;
                        let lo_fp32 = lo_bf16 << 16;
                        let hi_fp32 = hi_bf16 << 16;
                        acc[i] = lo_fp32 | (hi_fp32 << 32);
                    }
                    ctx.accumulator.write(*r, acc);
                } else {
                    // Generic path: use vector_convert (256-bit result)
                    let mode = super::VectorAlu::ctx_rounding(ctx);
                    let result = super::VectorAlu::vector_convert(&vec_data, from, to, mode);
                    // Pack into accumulator as raw u64 words
                    let mut acc = [0u64; 8];
                    for i in 0..4 {
                        acc[i] = result[i * 2] as u64 | ((result[i * 2 + 1] as u64) << 32);
                    }
                    ctx.accumulator.write(*r, acc);
                }
            }
            Some(dest) => {
                // Non-accumulator dest (vector register)
                let mode = super::VectorAlu::ctx_rounding(ctx);
                let result = super::VectorAlu::vector_convert(&vec_data, from, to, mode);
                let latency = load_latency_for_address(Self::get_address(op, ctx));
                ctx.queue_vector_load(dest.clone(), result, latency);
            }
            None => {}
        }

        Self::apply_post_modify(op, ctx, post_modify);
    }

    // ========== Fused compute+store handlers ==========

    /// Execute `vst.srs`: shift-round-saturate accumulator, store to memory.
    ///
    /// Reads accumulator data, applies SRS to produce a narrower vector,
    /// then stores the result to memory.
    fn execute_fused_store_srs(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        tile: &mut Tile,
        post_modify: &PostModify,
        neighbors: Option<&mut NeighborMemory>,
        view: Option<&NeighborView>,
    ) {
        let acc_reg = Self::get_acc_source(op);
        let shift = Self::get_shift_amount(op, ctx);
        let from = op.from_type.unwrap_or(ElementType::Int32);
        let to = op.element_type.unwrap_or(ElementType::Int16);

        // Half (bml/bmh) -> 512-bit single register, narrow SRS.
        // Full/None (cm) -> 1024-bit wide pair, SRS each half and pack.
        let is_half =
            matches!(op.accum_width, Some(crate::interpreter::decode::register_map::AccumWidth::Half));

        let addr = Self::get_store_address(op, ctx);

        if is_half {
            let acc = ctx.accumulator.read(acc_reg);
            let narrowed = super::VectorAlu::vector_srs_from_acc(&acc, shift, from, to, &ctx.srs_config);

            let to_bits = to.bits() as usize;
            let from_bits = from.bits() as usize;
            let lanes = if from_bits <= 32 { 16 } else { 8 };
            let store_bytes = lanes * to_bits / 8;

            log::debug!(
                "[FUSED] vst.srs: acc{} -> addr=0x{:X} ({} bytes, {:?}->{:?}, half)",
                acc_reg,
                addr,
                store_bytes,
                from,
                to,
            );

            let (quadrant, local_offset) = decode_data_address(addr);
            if quadrant == MemoryQuadrant::Local {
                ctx.record_core_bank_access(local_offset as u32, store_bytes, tile.bank_layout());
            }

            Self::write_vector_to_memory(tile, addr, narrowed, neighbors, view);
        } else {
            // Wide SRS: read Acc1024 (cm-register), SRS each half, pack results.
            let acc_wide = ctx.accumulator.read_wide(acc_reg);
            let acc_lo: [u64; 8] = acc_wide[..8].try_into().unwrap();
            let acc_hi: [u64; 8] = acc_wide[8..].try_into().unwrap();

            let result_lo = super::VectorAlu::vector_srs_from_acc(&acc_lo, shift, from, to, &ctx.srs_config);
            let result_hi = super::VectorAlu::vector_srs_from_acc(&acc_hi, shift, from, to, &ctx.srs_config);

            // Pack the two halves. For reduction SRS (e.g., Acc32->i8),
            // each half only fills a fraction of its 8-word result.
            let from_bits = from.bits() as usize;
            let to_bits = to.bits() as usize;
            let lanes_per_half = if from_bits <= 32 { 16 } else { 8 };
            let words_per_half = (lanes_per_half * to_bits + 31) / 32;
            let n = words_per_half.min(8);

            let mut packed = [0u32; 8];
            packed[..n].copy_from_slice(&result_lo[..n]);
            if n + n <= 8 {
                packed[n..n + n].copy_from_slice(&result_hi[..n]);
            }

            let store_bytes = n * 2 * 4; // Both halves, 4 bytes per u32 word

            log::debug!(
                "[FUSED] vst.srs: acc{} -> addr=0x{:X} ({} bytes, {:?}->{:?}, wide)",
                acc_reg,
                addr,
                store_bytes,
                from,
                to,
            );

            let (quadrant, local_offset) = decode_data_address(addr);
            if quadrant == MemoryQuadrant::Local {
                ctx.record_core_bank_access(local_offset as u32, store_bytes, tile.bank_layout());
            }

            Self::write_vector_to_memory(tile, addr, packed, neighbors, view);
        }

        Self::apply_post_modify(op, ctx, post_modify);
    }

    /// Execute `vst.pack`: pack vector, store narrowed data to memory.
    ///
    /// The source is a 512-bit x-register. Each 256-bit half is packed
    /// independently, then the two packed halves are concatenated into
    /// a single 256-bit (32-byte) store.
    fn execute_fused_store_pack(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        tile: &mut Tile,
        post_modify: &PostModify,
        neighbors: Option<&mut NeighborMemory>,
        view: Option<&NeighborView>,
    ) {
        let enc_name = op.encoding_name.as_deref().unwrap_or("");
        let (bits_i, bits_o, signed) = super::vector_pack::pack_widths_from_name(enc_name);

        // Read 512-bit x-register source (wl + wh).
        let src_lo = Self::get_vector_source(op, ctx);
        // Read the high half by incrementing the register index.
        let src_hi = {
            let mut hi = [0u32; 8];
            for src in &op.sources {
                if let Operand::VectorReg(r) = src {
                    hi = ctx.vector.read(r + 1);
                    break;
                }
            }
            hi
        };

        // vst.pack reads crSat (Uses = [crSat]); derive the narrow mode from the
        // live saturation register instead of assuming truncation. Matches the
        // standalone execute_pack path and real NPU1 silicon (saturating crSat
        // clamps int16->int8 rather than taking the low byte).
        let mode = super::vector_pack::PackMode::from_sat_flags(
            ctx.srs_config.saturate(),
            ctx.srs_config.symmetric_saturate(),
        );
        let packed_lo = super::vector_pack::pack_half(&src_lo, bits_i, bits_o, signed, mode);
        let packed_hi = super::vector_pack::pack_half(&src_hi, bits_i, bits_o, signed, mode);

        // Each half produces (256/bits_i * bits_o) bits of packed data.
        let words_per_half = ((256 / bits_i as usize) * bits_o as usize / 32).max(1);
        let mut result = [0u32; 8];
        let n = words_per_half.min(4); // Each half fits in at most 4 words
        result[..n].copy_from_slice(&packed_lo[..n]);
        result[n..n + n].copy_from_slice(&packed_hi[..n]);

        let addr = Self::get_store_address(op, ctx);
        log::debug!("[FUSED] vst.pack: addr=0x{:X} ({}-bit -> {}-bit)", addr, bits_i, bits_o,);

        let (quadrant, local_offset) = decode_data_address(addr);
        if quadrant == MemoryQuadrant::Local {
            let store_bytes = n * 2 * 4;
            ctx.record_core_bank_access(local_offset as u32, store_bytes, tile.bank_layout());
        }

        Self::write_vector_to_memory(tile, addr, result, neighbors, view);
        Self::apply_post_modify(op, ctx, post_modify);
    }

    /// Execute `vst.conv`: convert type (e.g., f32 -> bf16), store to memory.
    ///
    /// Reads a vector register, applies type conversion, then stores
    /// the converted result to memory.
    fn execute_fused_store_convert(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        tile: &mut Tile,
        post_modify: &PostModify,
        neighbors: Option<&mut NeighborMemory>,
        view: Option<&NeighborView>,
    ) {
        let from = op.from_type.unwrap_or(ElementType::Float32);
        let to = op.element_type.unwrap_or(ElementType::BFloat16);

        // vst.conv source can be AccumReg (bmh/bml) or VectorReg.
        // For fp32->bf16: the accumulator holds 16 fp32 values packed as
        // 8 u64 words (2 x f32 per u64). Extract them into [u32; 8] for
        // the convert path.
        let src = if let Some(acc_reg) = Self::find_acc_source(op) {
            let acc = ctx.accumulator.read(acc_reg);
            // Accumulator in float mode: each u64 = (hi_f32 << 32) | lo_f32.
            // Extract the 8 low f32 words. The high words go into the upper
            // half of the vector which we handle separately if needed.
            //
            // For bf16 output (16 values in 256 bits = 32 bytes), we need
            // all 16 f32 values. Pack them as 8 pairs into the convert path.
            if from == ElementType::Float32 && to == ElementType::BFloat16 {
                // Round-narrow bf16 from accumulator float data, per the
                // configured rounding mode (HW: crRnd governs to_v16bfloat16).
                let mode = super::VectorAlu::ctx_rounding(ctx);
                let mut result = [0u32; 8];
                for i in 0..8 {
                    let f_lo = f32::from_bits(acc[i] as u32);
                    let f_hi = f32::from_bits((acc[i] >> 32) as u32);
                    let bf_lo = super::vector_float::f32_to_bf16(f_lo, mode);
                    let bf_hi = super::vector_float::f32_to_bf16(f_hi, mode);
                    result[i] = (bf_lo as u32) | ((bf_hi as u32) << 16);
                }
                let addr = Self::get_store_address(op, ctx);
                log::debug!("[FUSED] vst.conv: addr=0x{:X} ({:?} -> {:?}, from accum)", addr, from, to,);
                let (quadrant, local_offset) = decode_data_address(addr);
                if quadrant == MemoryQuadrant::Local {
                    ctx.record_core_bank_access(local_offset as u32, 32, tile.bank_layout());
                }
                Self::write_vector_to_memory(tile, addr, result, neighbors, view);
                Self::apply_post_modify(op, ctx, post_modify);
                return;
            }
            // Non-bf16 path: extract low 32 bits of each acc word
            let mut v = [0u32; 8];
            for i in 0..8 {
                v[i] = acc[i] as u32;
            }
            v
        } else {
            Self::get_vector_source(op, ctx)
        };

        let mode = super::VectorAlu::ctx_rounding(ctx);
        let converted = super::VectorAlu::vector_convert(&src, from, to, mode);

        let addr = Self::get_store_address(op, ctx);
        log::debug!("[FUSED] vst.conv: addr=0x{:X} ({:?} -> {:?})", addr, from, to,);

        let (quadrant, local_offset) = decode_data_address(addr);
        if quadrant == MemoryQuadrant::Local {
            ctx.record_core_bank_access(local_offset as u32, 32, tile.bank_layout());
        }

        Self::write_vector_to_memory(tile, addr, converted, neighbors, view);
        Self::apply_post_modify(op, ctx, post_modify);
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
        view: Option<&NeighborView>,
    ) {
        // Use get_store_address which searches all sources for Memory operand.
        // For vst, sources[0] is the vector data, Memory is sources[1].
        let addr = Self::get_store_address(op, ctx);
        log::debug!("[VST] addr=0x{:X} sources={:?} dest={:?}", addr, op.sources, op.dest);

        // Track bank access for conflict detection (local memory only)
        let (quadrant, local_offset) = decode_data_address(addr);
        if quadrant == MemoryQuadrant::Local {
            ctx.record_core_bank_access(local_offset as u32, 32, tile.bank_layout());
        }

        // Get data from sources or dest. Check VectorReg, AccumReg, or ControlReg.
        let vec_data = Self::read_store_data_wide(op, ctx);

        if let Some(ref data) = vec_data {
            log::trace!("[VST] writing to addr=0x{:X} mem_width={:?}", addr, op.mem_width);
            if crate::debug::watch::is_watched(
                addr as u64,
                if op.mem_width == MemWidth::QuadWord {
                    16
                } else {
                    32
                },
            ) {
                crate::debug::watch::log_core_store(
                    ctx.cycles,
                    tile.col,
                    tile.row,
                    ctx.pc(),
                    addr as u64,
                    data[0],
                );
            }
            if op.mem_width == MemWidth::QuadWord {
                let half = [data[0], data[1], data[2], data[3], 0, 0, 0, 0];
                Self::write_vector_to_memory(tile, addr, half, neighbors, view);
            } else {
                Self::write_vector_to_memory(tile, addr, *data, neighbors, view);
            }
        } else {
            log::warn!("[VST] no data register found! sources={:?} dest={:?}", op.sources, op.dest);
        }

        // Apply post-modify (vector width = 32 bytes)
        Self::apply_post_modify(op, ctx, post_modify);
    }

    /// Read 256-bit store data from the first VectorReg, AccumReg, or ControlReg
    /// found in the op's sources or dest. Returns [u32; 8] or None.
    fn read_store_data_wide(op: &SlotOp, ctx: &ExecutionContext) -> Option<[u32; 8]> {
        use crate::interpreter::decode::register_map::AccumWidth;

        for operand in op.sources.iter().chain(op.dest.iter()) {
            match operand {
                // Store data is a NoBypass consumer: it never receives ALU
                // forwarding, so a vector producer's result is visible only at
                // its full architectural latency (the bf16 split-tile store
                // reads the OLD value the bundle after a `VMOV x,bml`).
                Operand::VectorReg(r) => return Some(ctx.vector.read_store(*r)),
                Operand::AccumReg(r) => {
                    // Read the right quarter/half based on accum_width.
                    let accum = ctx.accumulator.read(*r);
                    let width = op.accum_width.unwrap_or(AccumWidth::Half);
                    let lanes: &[u64] = match width {
                        AccumWidth::QuarterLow => &accum[0..4],
                        AccumWidth::QuarterHigh => &accum[4..8],
                        AccumWidth::Half | AccumWidth::Full => &accum[0..4],
                    };
                    // Convert [u64] lanes to [u32; 8] for memory write.
                    let mut result = [0u32; 8];
                    for (i, &lane) in lanes.iter().enumerate() {
                        result[i * 2] = lane as u32;
                        result[i * 2 + 1] = (lane >> 32) as u32;
                    }
                    return Some(result);
                }
                Operand::ControlReg(id) if *id >= 16 && *id <= 19 => {
                    // q0-q3 mask registers: full 128-bit read.
                    let q_idx = (*id - 16) as u8;
                    let mask = ctx.mask.read(q_idx);
                    return Some([mask[0], mask[1], mask[2], mask[3], 0, 0, 0, 0]);
                }
                Operand::ControlReg(id) if *id >= 28 && *id <= 31 => {
                    // ql0-ql3: low 64 bits of mask register.
                    let q_idx = (*id - 28) as u8;
                    let mask = ctx.mask.read(q_idx);
                    return Some([mask[0], mask[1], 0, 0, 0, 0, 0, 0]);
                }
                Operand::ControlReg(id) if *id >= 32 && *id <= 35 => {
                    // qh0-qh3: high 64 bits of mask register.
                    let q_idx = (*id - 32) as u8;
                    let mask = ctx.mask.read(q_idx);
                    return Some([mask[2], mask[3], 0, 0, 0, 0, 0, 0]);
                }
                _ => {}
            }
        }
        None
    }

    /// Get the pointer register from a slot op's source operands.
    /// Searches all sources, not just the first, because stores have
    /// the data source before the address operand.
    fn get_pointer_reg(op: &SlotOp) -> Option<u8> {
        for src in &op.sources {
            match src {
                Operand::Memory { base, .. } => return Some(*base),
                Operand::PointerReg(r) => return Some(*r),
                _ => {}
            }
        }
        None
    }

    /// Get address from memory operand or pointer register.
    ///
    /// Three operand layouts are handled:
    /// 1. `Memory { base, offset }` -- pre-scaled byte offset from AG decode
    /// 2. `[PointerReg, ModifierReg]` -- indexed register (byte offset in modifier)
    /// 3. `[PointerReg, Immediate/ScalarReg]` -- word-scaled offset
    ///
    /// Exposed `pub(crate)` so cycle-accurate accounting (bank-conflict
    /// tracking and watchpoint matching in `record_memory_access`) can use
    /// the same effective address the load actually accesses, instead of
    /// re-deriving an approximation that drops the modifier-register offset.
    pub(crate) fn get_address(op: &SlotOp, ctx: &ExecutionContext) -> u32 {
        // Search all sources for Memory operand (encapsulates ptr+offset, already scaled).
        // For stores, the data source may precede the address operand.
        for src in &op.sources {
            if let Operand::Memory { base, offset } = src {
                let base_addr = ctx.pointer_read(*base);
                let result = base_addr.wrapping_add(*offset as i32 as u32);
                return result;
            }
        }

        // Register-indirect addressing. The base is a PointerReg; locate it by
        // SEARCH rather than position, because fused compute-loads/stores
        // (vlda.ups/vst.srs/...) carry the shift ScalarReg *before* the pointer
        // (sources = [shift, p0, ...]). Any operand following the pointer is the
        // offset/index (ModifierReg for indexed, ScalarReg/Immediate otherwise);
        // a post-increment carries no offset operand (the step lives in
        // post_modify and is applied separately).
        if let Some(ptr_idx) = op.sources.iter().position(|s| matches!(s, Operand::PointerReg(_))) {
            let base_addr = match &op.sources[ptr_idx] {
                Operand::PointerReg(r) => ctx.pointer_read(*r),
                _ => unreachable!(),
            };
            let offset = op.sources.get(ptr_idx + 1).map_or(0, |src| match src {
                Operand::Immediate(v) => (*v as i32 * 4) as u32, // Scale by word size
                Operand::ScalarReg(r) => ctx.scalar_read(*r).wrapping_mul(4),
                // Modifier registers contain byte offsets (set via mov dj0/m0, rN)
                Operand::ModifierReg(r) => ctx.modifier_read(*r),
                _ => 0, // post-increment / no offset operand
            });
            return base_addr.wrapping_add(offset);
        }

        // No pointer: legacy positional fallback (immediate/scalar base).
        let base_addr = op.sources.first().map_or(0, |src| match src {
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
    ///
    /// Exposed `pub(crate)` for the same reason as [`get_address`]: bank-conflict
    /// tracking and watchpoint matching call this from
    /// `CycleAccurateExecutor::record_memory_access` so the recorded address
    /// matches the address the store actually writes to.
    pub(crate) fn get_store_address(op: &SlotOp, ctx: &ExecutionContext) -> u32 {
        // First check for Memory operand anywhere (encapsulates ptr+offset)
        for src in &op.sources {
            if let Operand::Memory { base, offset } = src {
                let base_addr = ctx.pointer_read(*base);
                return base_addr.wrapping_add(*offset as i32 as u32);
            }
        }

        // Sources[0] is a pointer: either bare `[pN]` or indexed `[pN, djM]`.
        // If sources[1] is a modifier register, include it as a byte offset.
        if let Some(Operand::PointerReg(r)) = op.sources.first() {
            let base_addr = ctx.pointer_read(*r);
            let offset = op.sources.get(1).map_or(0u32, |src| match src {
                Operand::ModifierReg(m) => ctx.modifier_read(*m),
                Operand::Immediate(v) => *v as u32,
                _ => 0,
            });
            return base_addr.wrapping_add(offset);
        }

        // Fused stores (vst.srs/vst.pack/...) carry the pointer AFTER the data
        // and shift operands (sources = [acc, shift, pointer]); a
        // post-increment carries no offset operand (the step lives in
        // post_modify). Locate the pointer by SEARCH -- mirroring get_address --
        // so the positional fallback below never mistakes the shift register for
        // the base address. Without this, a post-increment `vst.srs [pN], #imm`
        // read the shift (e.g. 4) as the store address and scattered output to
        // ~0x4; indexed forms decode to `Memory{}` and were unaffected, so it
        // surfaced only on compiles that picked post-increment addressing.
        if let Some(ptr_idx) = op.sources.iter().position(|s| matches!(s, Operand::PointerReg(_))) {
            let base_addr = match &op.sources[ptr_idx] {
                Operand::PointerReg(r) => ctx.pointer_read(*r),
                _ => unreachable!(),
            };
            let offset = op.sources.get(ptr_idx + 1).map_or(0, |src| match src {
                Operand::ModifierReg(r) => ctx.modifier_read(*r),
                Operand::Immediate(v) => (*v as i32 * 4) as u32,
                Operand::ScalarReg(r) => ctx.scalar_read(*r).wrapping_mul(4),
                _ => 0,
            });
            return base_addr.wrapping_add(offset);
        }

        // Kernel layout: sources[0]=value, sources[1]=pointer, sources[2]=index
        let base_addr = op.sources.get(1).map_or(0, |src| match src {
            Operand::PointerReg(r) => ctx.pointer_read(*r),
            Operand::ScalarReg(r) => ctx.scalar_read(*r),
            Operand::Immediate(v) => *v as u32,
            other => {
                log::warn!(
                    "[MEMORY] get_store_address: unexpected base operand {:?}, defaulting to 0",
                    other
                );
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
                log::warn!(
                    "[MEMORY] get_store_address: unexpected offset operand {:?}, defaulting to 0",
                    other
                );
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
            op.sources
                .first()
                .and_then(|first| match first {
                    Operand::PointerReg(_) | Operand::Memory { .. } => op.dest.as_ref(),
                    _ => Some(first),
                })
                .or(op.dest.as_ref())
        };

        Self::read_store_operand(operand, ctx, width)
    }

    /// Read a store value from an operand reference.
    /// Extract the source register operand for a store instruction.
    ///
    /// Uses the same logic as `get_store_value` but returns the Operand
    /// instead of reading it, for use with deferred partial-word stores.
    fn get_store_source_operand(op: &SlotOp) -> Operand {
        // For partial-word stores, the data register is the first non-address
        // operand. The LLVM FFI decoder puts the data register in op.dest when
        // it appears first in the MCInst defs, so fall back to dest when the
        // sources contain only address operands (PointerReg, Memory, ModifierReg).
        let operand = op
            .sources
            .first()
            .and_then(|first| match first {
                Operand::PointerReg(_) | Operand::Memory { .. } | Operand::ModifierReg(_) => op.dest.clone(),
                _ => Some(first.clone()),
            })
            .or_else(|| op.dest.clone());
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
                // Store data: NoBypass consumer (see read_store_data_wide).
                let vec = ctx.vector.read_store(*r);
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
    ///
    /// When the processor bus is enabled (Core_Processor_Bus register),
    /// addresses in the register space (0x10000-0x3FFFF, CardDir 1-3)
    /// read tile configuration registers instead of data memory.
    fn read_memory(
        tile: &Tile,
        addr: u32,
        width: MemWidth,
        neighbors: Option<&mut NeighborMemory>,
        view: Option<&NeighborView>,
    ) -> u64 {
        // Processor bus: the core accesses tile configuration registers via
        // a window at base 0x80000 in the core's address space. Register
        // offset = (address - PROC_BUS_BASE). This maps to the tile's memory
        // module, core module, etc. registers at their AM025 offsets.
        // Window constants from xdna_archspec::aie2::memory_map.
        if tile.processor_bus_enabled && addr >= memory_map::PROC_BUS_BASE && addr < memory_map::PROC_BUS_END
        {
            let reg_offset = addr - memory_map::PROC_BUS_BASE;
            let reg_val = tile.read_register_pure(reg_offset);
            log::debug!("[PROC_BUS] read addr=0x{:X} reg=0x{:X} -> 0x{:08X}", addr, reg_offset, reg_val);
            return reg_val as u64;
        }

        let (quadrant, offset) = decode_data_address(addr);

        // Select memory source based on quadrant
        let mem: &[u8] = if quadrant == MemoryQuadrant::Local || neighbors.is_none() {
            // Fast path: local tile memory
            tile.data_memory()
        } else {
            // Cross-tile read: if a view was passed (lazy mode), refresh
            // the snapshot for this quadrant before consulting it. Without
            // a view (eager mode), the caller has already pre-populated.
            let n = neighbors.unwrap();
            if let Some(v) = view {
                n.ensure_snapshot(quadrant, v);
            }
            match n.get_memory(quadrant) {
                Some(m) => m,
                None => {
                    log::trace!("[LOAD] cross-tile read to non-existent neighbor {:?}", quadrant);
                    return 0;
                }
            }
        };

        Self::read_from_slice(mem, offset, width)
    }

    /// Read a value from a memory slice at the given offset.
    fn read_from_slice(mem: &[u8], addr: usize, width: MemWidth) -> u64 {
        if addr >= mem.len() {
            log::warn!("[MEMORY] OOB read: addr=0x{:X} width={:?} mem_size=0x{:X}", addr, width, mem.len());
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
                    u32::from_le_bytes([mem[addr], mem[addr + 1], mem[addr + 2], mem[addr + 3]]) as u64
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
    ///
    /// When the processor bus is enabled, writes to the register space
    /// (0x10000-0x3FFFF) target tile configuration registers.
    pub fn write_memory(
        tile: &mut Tile,
        addr: u32,
        value: u64,
        width: MemWidth,
        neighbors: Option<&mut NeighborMemory>,
        // Writes go to NeighborMemory's pending_writes buffer; they don't
        // need to refresh a snapshot first. `view` is accepted for symmetry
        // with the read paths (callers thread the same set of args through).
        _view: Option<&NeighborView>,
    ) {
        // Processor bus: write to tile config registers via the 0x80000 window.
        // Window constants from xdna_archspec::aie2::memory_map.
        if tile.processor_bus_enabled && addr >= memory_map::PROC_BUS_BASE && addr < memory_map::PROC_BUS_END
        {
            let reg_offset = addr - memory_map::PROC_BUS_BASE;
            log::debug!("[PROC_BUS] write addr=0x{:X} reg=0x{:X} value=0x{:X}", addr, reg_offset, value);
            tile.registers.insert(reg_offset, value as u32);
            return;
        }

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
    fn write_dest_with_latency(
        op: &SlotOp,
        ctx: &mut ExecutionContext,
        value: u64,
        width: MemWidth,
        latency: u64,
    ) {
        if let Some(dest) = &op.dest {
            match dest {
                Operand::ScalarReg(_) | Operand::PointerReg(_) | Operand::ModifierReg(_) => {
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
    pub fn read_vector_from_memory(
        tile: &Tile,
        addr: u32,
        neighbors: Option<&mut NeighborMemory>,
        view: Option<&NeighborView>,
    ) -> [u32; 8] {
        let (quadrant, offset) = decode_data_address(addr);

        // Select memory source
        let mem: &[u8] = if quadrant == MemoryQuadrant::Local || neighbors.is_none() {
            tile.data_memory()
        } else {
            let n = neighbors.unwrap();
            if let Some(v) = view {
                n.ensure_snapshot(quadrant, v);
            }
            match n.get_memory(quadrant) {
                Some(m) => m,
                None => {
                    log::trace!("[VLOAD] cross-tile read to non-existent neighbor {:?}", quadrant);
                    return [0u32; 8];
                }
            }
        };

        Self::read_vector_from_slice(mem, offset)
    }

    /// Read 8 x u32 from a memory slice at the given byte offset.
    fn read_vector_from_slice(mem: &[u8], addr: usize) -> [u32; 8] {
        let mut result = [0u32; 8];

        if addr + 31 < mem.len() {
            for i in 0..8 {
                let offset = addr + i * 4;
                result[i] =
                    u32::from_le_bytes([mem[offset], mem[offset + 1], mem[offset + 2], mem[offset + 3]]);
            }
        }

        result
    }

    /// Write a vector to memory (256 bits = 32 bytes) with cross-tile routing.
    pub fn write_vector_to_memory(
        tile: &mut Tile,
        addr: u32,
        value: [u32; 8],
        neighbors: Option<&mut NeighborMemory>,
        // See write_memory: writes don't need a snapshot refresh.
        _view: Option<&NeighborView>,
    ) {
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
    use xdna_archspec::aie2::isa::SemanticOp;

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

        assert!(MemoryUnit::execute(&op, &mut ctx, &mut tile, None, None));
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

        MemoryUnit::execute(&op, &mut ctx, &mut tile, None, None);
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

        MemoryUnit::execute(&op, &mut ctx, &mut tile, None, None);
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

        MemoryUnit::execute(&op, &mut ctx, &mut tile, None, None);
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

        MemoryUnit::execute(&op, &mut ctx, &mut tile, None, None);
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

        MemoryUnit::execute(&op, &mut ctx, &mut tile, None, None);
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

        MemoryUnit::execute(&op, &mut ctx, &mut tile, None, None);
        ctx.flush_pending_writes();
        assert_eq!(ctx.scalar.read(0), 0xABCD);
    }

    #[test]
    fn test_vector_memory_helpers() {
        let mut tile = make_tile();

        let data = [1, 2, 3, 4, 5, 6, 7, 8];
        MemoryUnit::write_vector_to_memory(&mut tile, 0x300, data, None, None);

        let read_back = MemoryUnit::read_vector_from_memory(&tile, 0x300, None, None);
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
        MemoryUnit::write_vector_to_memory(&mut tile, 0x400, test_data, None, None);

        // Set pointer
        ctx.pointer.write(0, 0x400);

        // Load vector: v0 = [p0]
        let op = SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load)
            .with_mem_width(MemWidth::Vector256)
            .with_dest(Operand::VectorReg(0))
            .with_source(Operand::PointerReg(0));

        MemoryUnit::execute(&op, &mut ctx, &mut tile, None, None);
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

        MemoryUnit::execute(&op, &mut ctx, &mut tile, None, None);

        // Verify all 8 lanes were stored correctly
        let stored = MemoryUnit::read_vector_from_memory(&tile, 0x500, None, None);
        assert_eq!(stored, test_data);
    }

    #[test]
    fn test_vector256_load_with_post_modify() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        let test_data = [1, 2, 3, 4, 5, 6, 7, 8];
        MemoryUnit::write_vector_to_memory(&mut tile, 0x600, test_data, None, None);
        ctx.pointer.write(0, 0x600);

        // Load with post-modify: v0 = [p0], p0 += 32
        let op = SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load)
            .with_mem_width(MemWidth::Vector256)
            .with_post_modify(PostModify::Immediate(32)) // 256 bits = 32 bytes
            .with_dest(Operand::VectorReg(0))
            .with_source(Operand::PointerReg(0));

        MemoryUnit::execute(&op, &mut ctx, &mut tile, None, None);
        ctx.flush_pending_writes();

        assert_eq!(ctx.vector.read(0), test_data);
        assert_eq!(ctx.pointer.read(0), 0x620); // 0x600 + 32
    }

    #[test]
    fn test_vector_load_a() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Write test vector data
        let test_data =
            [0xAABBCCDD, 0x11223344, 0x55667788, 0x99AABBCC, 0xDDEEFF00, 0x12345678, 0x9ABCDEF0, 0xFEDCBA98];
        MemoryUnit::write_vector_to_memory(&mut tile, 0x700, test_data, None, None);
        ctx.pointer.write(0, 0x700);

        // VLDA: v2 = [p0]
        let op = SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load)
            .as_vector(ElementType::Int32)
            .with_mem_width(MemWidth::Vector256)
            .with_dest(Operand::VectorReg(2))
            .with_source(Operand::PointerReg(0));

        assert!(MemoryUnit::execute(&op, &mut ctx, &mut tile, None, None));
        ctx.flush_pending_writes();
        assert_eq!(ctx.vector.read(2), test_data);
        assert_eq!(ctx.pointer.read(0), 0x700); // No post-modify
    }

    #[test]
    fn test_vector_load_a_with_post_modify() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        let test_data = [1, 2, 3, 4, 5, 6, 7, 8];
        MemoryUnit::write_vector_to_memory(&mut tile, 0x800, test_data, None, None);
        ctx.pointer.write(1, 0x800);

        // VLDA: v0 = [p1], p1 += 32
        let op = SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load)
            .as_vector(ElementType::Int32)
            .with_mem_width(MemWidth::Vector256)
            .with_post_modify(PostModify::Immediate(32))
            .with_dest(Operand::VectorReg(0))
            .with_source(Operand::PointerReg(1));

        MemoryUnit::execute(&op, &mut ctx, &mut tile, None, None);
        ctx.flush_pending_writes();
        assert_eq!(ctx.vector.read(0), test_data);
        assert_eq!(ctx.pointer.read(1), 0x820); // 0x800 + 32
    }

    #[test]
    fn test_vector_load_b() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        let test_data =
            [0xDEADBEEF, 0xCAFEBABE, 0x12345678, 0x9ABCDEF0, 0x0F0F0F0F, 0xF0F0F0F0, 0xAAAA5555, 0x5555AAAA];
        MemoryUnit::write_vector_to_memory(&mut tile, 0x900, test_data, None, None);
        ctx.pointer.write(4, 0x900); // B-channel typically uses p4-p7

        // VLDB: v3 = [p4]
        let op = SlotOp::from_semantic(SlotIndex::LoadB, SemanticOp::Load)
            .as_vector(ElementType::Int32)
            .with_mem_width(MemWidth::Vector256)
            .with_dest(Operand::VectorReg(3))
            .with_source(Operand::PointerReg(4));

        assert!(MemoryUnit::execute(&op, &mut ctx, &mut tile, None, None));
        ctx.flush_pending_writes();
        assert_eq!(ctx.vector.read(3), test_data);
    }

    #[test]
    fn test_vector_load_b_with_modifier_register() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        let test_data = [10, 20, 30, 40, 50, 60, 70, 80];
        MemoryUnit::write_vector_to_memory(&mut tile, 0xA00, test_data, None, None);
        ctx.pointer.write(5, 0xA00);
        ctx.modifier.write(2, 64); // m2 = 64

        // VLDB: v1 = [p5], p5 += m2
        let op = SlotOp::from_semantic(SlotIndex::LoadB, SemanticOp::Load)
            .as_vector(ElementType::Int32)
            .with_mem_width(MemWidth::Vector256)
            .with_post_modify(PostModify::Register(2))
            .with_dest(Operand::VectorReg(1))
            .with_source(Operand::PointerReg(5));

        MemoryUnit::execute(&op, &mut ctx, &mut tile, None, None);
        ctx.flush_pending_writes();
        assert_eq!(ctx.vector.read(1), test_data);
        assert_eq!(ctx.pointer.read(5), 0xA40); // 0xA00 + 64
    }

    #[test]
    fn test_vector_store() {
        let mut ctx = make_ctx();
        let mut tile = make_tile();

        let test_data =
            [0x11111111, 0x22222222, 0x33333333, 0x44444444, 0x55555555, 0x66666666, 0x77777777, 0x88888888];
        ctx.vector.write(4, test_data);
        ctx.pointer.write(2, 0xB00);

        // VST: [p2] = v4
        let op = SlotOp::from_semantic(SlotIndex::Store, SemanticOp::Store)
            .as_vector(ElementType::Int32)
            .with_mem_width(MemWidth::Vector256)
            .with_dest(Operand::VectorReg(4))
            .with_source(Operand::PointerReg(2));

        assert!(MemoryUnit::execute(&op, &mut ctx, &mut tile, None, None));

        let stored = MemoryUnit::read_vector_from_memory(&tile, 0xB00, None, None);
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

        MemoryUnit::execute(&op, &mut ctx, &mut tile, None, None);

        let stored = MemoryUnit::read_vector_from_memory(&tile, 0xC00, None, None);
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

        assert!(MemoryUnit::execute(&op, &mut ctx, &mut tile, None, None));
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

        MemoryUnit::execute(&op, &mut ctx, &mut tile, None, None);
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

        MemoryUnit::execute(&op, &mut ctx, &mut tile, None, None);
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
        nbr.drain_writes(&mut device);

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
        let value = MemoryUnit::read_memory(&tile, 0x50100, MemWidth::Word, Some(&mut nbr), None);
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
        let result = MemoryUnit::read_vector_from_memory(&tile, 0x60200, Some(&mut nbr), None);
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
        MemoryUnit::write_memory(&mut tile, 0x40400, 0xBAAD_F00D, MemWidth::Word, Some(&mut nbr), None);

        // Should be buffered, not yet in the south neighbor
        let south = device.tile(1, 2).unwrap();
        let mem = south.data_memory();
        let value = u32::from_le_bytes([mem[0x400], mem[0x401], mem[0x402], mem[0x403]]);
        assert_eq!(value, 0, "Write should be buffered, not applied yet");

        // Apply writes
        assert!(nbr.has_pending_writes());
        nbr.drain_writes(&mut device);

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
        MemoryUnit::write_vector_to_memory(&mut tile, 0x50800, test_data, Some(&mut nbr), None);

        // Apply and verify
        nbr.drain_writes(&mut device);

        let west = device.tile(0, 3).unwrap();
        let mem = west.data_memory();
        for (i, &expected) in test_data.iter().enumerate() {
            let offset = 0x800 + i * 4;
            let actual = u32::from_le_bytes([mem[offset], mem[offset + 1], mem[offset + 2], mem[offset + 3]]);
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
        let value = MemoryUnit::read_memory(&tile, 0x70100, MemWidth::Word, Some(&mut nbr), None);
        assert_eq!(value, 0x42424242);

        // Write to local memory (CardDir 7) with neighbors present
        MemoryUnit::write_memory(&mut tile, 0x70200, 0x99887766, MemWidth::Word, Some(&mut nbr), None);
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
        let value = MemoryUnit::read_memory(&tile, 0x50100, MemWidth::Word, Some(&mut nbr), None);
        assert_eq!(value, 0, "Read from non-existent west neighbor should return 0");
    }

    /// Lazy refresh: `read_memory` for a Local-quadrant address must NOT call
    /// `ensure_snapshot`, even when both `neighbors` and `view` are passed.
    /// Local reads are the common case; making them snapshot would cost
    /// 1 gen-check per access on every load instruction.
    #[test]
    fn read_memory_local_addr_does_not_snapshot() {
        let mut device = crate::device::DeviceState::new_npu1();
        let mut nbr = NeighborMemory::new(1, 3);
        let mut tile = make_tile();
        tile.write_data_u32(0x100, 0xCAFE_F00D);

        let (_own, view) = device.split_tile_mut(1, 3).expect("valid coords");

        // CardDir 7 = Local
        let value = MemoryUnit::read_memory(&tile, 0x70100, MemWidth::Word, Some(&mut nbr), Some(&view));
        assert_eq!(value, 0xCAFE_F00D);
        assert_eq!(nbr.ensure_snapshot_calls, 0, "local reads must not consult the view");
    }

    /// Lazy refresh: `read_memory` for a cross-tile address with `view=Some`
    /// must call `ensure_snapshot` exactly once for the accessed quadrant
    /// (and zero times for the other three -- this is the whole point of
    /// lazy access: pay only for what you read).
    #[test]
    fn read_memory_cross_tile_snapshots_once_for_accessed_quadrant() {
        let mut device = crate::device::DeviceState::new_npu1();
        // West neighbor of (1,3) is (0,3). Seed it.
        device.tile_mut(0, 3).unwrap().data_memory_mut()[0x100..0x104]
            .copy_from_slice(&0xABCD_1234u32.to_le_bytes());

        let mut nbr = NeighborMemory::new(1, 3);
        let tile = Tile::compute(1, 3);
        let (_own, view) = device.split_tile_mut(1, 3).expect("valid coords");

        // CardDir 5 = West, offset 0x100
        let value = MemoryUnit::read_memory(&tile, 0x50100, MemWidth::Word, Some(&mut nbr), Some(&view));
        assert_eq!(value, 0xABCD_1234);
        assert_eq!(nbr.ensure_snapshot_calls, 1, "only the accessed quadrant should be snapshotted");
    }

    /// Eager-mode compatibility: when `view=None`, `read_memory` must NOT
    /// call `ensure_snapshot` (caller is responsible for pre-populating
    /// the cache). This is the path the tests above use, and it's also
    /// the path through which existing callers can adopt the new
    /// signature without changing their snapshot policy.
    #[test]
    fn read_memory_view_none_does_not_snapshot() {
        let mut device = crate::device::DeviceState::new_npu1();
        device.tile_mut(0, 3).unwrap().data_memory_mut()[0x100..0x104]
            .copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());

        let mut nbr = NeighborMemory::new(1, 3);
        nbr.ensure_snapshot(MemoryQuadrant::West, &device);
        let pre_calls = nbr.ensure_snapshot_calls;

        let tile = Tile::compute(1, 3);
        let value = MemoryUnit::read_memory(&tile, 0x50100, MemWidth::Word, Some(&mut nbr), None);
        assert_eq!(value, 0xDEAD_BEEF);
        assert_eq!(nbr.ensure_snapshot_calls, pre_calls, "view=None must not trigger any new snapshot calls");
    }

    #[test]
    fn test_load_latency_uniform_for_all_quadrants() {
        // On AIE2, core loads have the same pipeline latency for ALL data
        // memory quadrants. The core accesses neighbor memory through direct
        // ports, not the stream switch, so there is no cross-tile routing
        // penalty for core load/store operations.

        // Local memory (CardDir 7 = East = local)
        assert_eq!(load_latency_for_address(0x70000), LATENCY_MEMORY as u64);
        assert_eq!(load_latency_for_address(0x7FFFF), LATENCY_MEMORY as u64);

        // West neighbor (CardDir 5): same latency as local
        assert_eq!(load_latency_for_address(0x50000), LATENCY_MEMORY as u64);
        assert_eq!(load_latency_for_address(0x5FFFF), LATENCY_MEMORY as u64);

        // North neighbor (CardDir 6): same latency as local
        assert_eq!(load_latency_for_address(0x60000), LATENCY_MEMORY as u64);

        // South neighbor (CardDir 4): same latency as local
        assert_eq!(load_latency_for_address(0x40000), LATENCY_MEMORY as u64);

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
        let value = MemoryUnit::read_memory(&tile, 0x70440, MemWidth::Word, None, None);
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
        MemoryUnit::write_memory(&mut tile, store_addr, p7_val as u64, MemWidth::Word, None, None);

        // Verify the value is in memory
        assert_eq!(tile.read_data_u32(local_offset), Some(0x78000));

        // Now clobber p7 (simulating mov p7, sp in the prologue)
        ctx.pointer.write(7, ctx.sp());
        assert_eq!(ctx.pointer.read(7), 0x70060);

        // Load p7 back from [sp - 32] using the Memory operand
        let load_op = SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load)
            .with_dest(Operand::PointerReg(7))
            .with_source(Operand::Memory { base: SP_PTR_INDEX, offset: -32 });

        MemoryUnit::execute(&load_op, &mut ctx, &mut tile, None, None);
        ctx.flush_pending_writes();

        assert_eq!(
            ctx.pointer.read(7),
            0x78000,
            "p7 should be restored to original value after load from [sp, #-32]"
        );
    }

    #[test]
    fn test_sp_relative_load_with_latency() {
        // Same as above but verify the load goes through the pending write
        // queue with proper latency (not flushed immediately).
        use crate::interpreter::state::SP_PTR_INDEX;
        use crate::interpreter::timing::LATENCY_MEMORY;

        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Explicitly zero p7 so we can verify the load hasn't committed yet.
        ctx.pointer.write(7, 0);

        ctx.set_sp(0x70080);
        ctx.cycles = 100;

        // Write test value to [sp - 32] = 0x70060 -> local 0x0060
        let addr = ctx.sp().wrapping_add(-32_i32 as u32);
        MemoryUnit::write_memory(&mut tile, addr, 0x78000, MemWidth::Word, None, None);

        // Load from [sp, #-32] into p7
        let load_op = SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load)
            .with_dest(Operand::PointerReg(7))
            .with_source(Operand::Memory { base: SP_PTR_INDEX, offset: -32 });

        MemoryUnit::execute(&load_op, &mut ctx, &mut tile, None, None);

        // Should NOT be committed yet (load latency = 7)
        assert_eq!(ctx.pointer.read(7), 0, "Load should not be committed immediately (latency pending)");

        // Before ready_cycle: read returns old value
        ctx.cycles = 101;
        assert_eq!(ctx.pointer_read(7), 0, "Before ready_cycle, should return old register value");

        // At ready_cycle: forward works, then commit
        ctx.cycles = 100 + LATENCY_MEMORY as u64;
        assert_eq!(ctx.pointer_read(7), 0x78000, "At ready_cycle, forward should return pending load value");
        ctx.commit_pending_writes();
        assert_eq!(ctx.pointer.read(7), 0x78000, "After commit, p7 should have restored value");
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

        MemoryUnit::execute(&store_op, &mut ctx, &mut tile, None, None);

        // Verify: [sp - 8] = 0x70040 - 8 = 0x70038 -> local 0x0038
        assert_eq!(
            tile.read_data_u32(0x38),
            Some(0xDEAD_BEEF),
            "Store via SP-relative addressing should write to correct local offset"
        );
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
        MemoryUnit::write_memory(
            &mut tile,
            store_addr,
            ctx.pointer_read(7) as u64,
            MemWidth::Word,
            None,
            None,
        );

        // Cycle 11: mov p7, sp (clobber)
        ctx.cycles = 11;
        ctx.queue_pointer_write(7, ctx.sp(), 1); // ready=12

        // Verify p7 is still old (deferred by 1)
        assert_eq!(ctx.pointer.read(7), 0x78000, "p7 live should still be old value (write deferred)");
        // But forward returns new value
        ctx.cycles = 12;
        assert_eq!(ctx.pointer_read(7), 0x70060, "p7 forward should return sp value after clobber");

        // Commit the clobber
        ctx.commit_pending_writes();
        assert_eq!(ctx.pointer.read(7), 0x70060, "p7 committed to sp value");

        // ... function body ...

        // Cycle 50: Restore p7 from [sp, #-32]
        ctx.cycles = 50;
        let load_op = SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load)
            .with_dest(Operand::PointerReg(7))
            .with_source(Operand::Memory { base: SP_PTR_INDEX, offset: -32 });
        MemoryUnit::execute(&load_op, &mut ctx, &mut tile, None, None);

        // Cycle 51: load not ready -- read returns clobbered sp value
        ctx.cycles = 51;
        assert_eq!(ctx.pointer_read(7), 0x70060, "Before ready_cycle, should read clobbered sp value");

        // Cycle 55: still not ready
        ctx.cycles = 55;
        assert_eq!(ctx.pointer_read(7), 0x70060, "Still before ready_cycle, should read clobbered sp value");

        // Cycle 57: load ready -- forward returns restored value, then commit
        ctx.cycles = 57;
        assert_eq!(
            ctx.pointer_read(7),
            0x78000,
            "At ready_cycle, forwarding should return restored p7 value"
        );
        ctx.commit_pending_writes();
        assert_eq!(ctx.pointer.read(7), 0x78000, "p7 should be committed to restored value");
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

        MemoryUnit::execute(&load_op, &mut ctx, &mut tile, None, None);

        // SP should be post-modified: 0x70100 + 4 = 0x70104
        assert_eq!(ctx.sp(), 0x70104, "SP should be post-modified by +4");

        // p7 should be UNAFFECTED
        assert_eq!(ctx.pointer.read(7), 0xBEEF, "p7 must not be affected by SP post-modify");

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

        MemoryUnit::execute(&op, &mut ctx, &mut tile, None, None);

        // Memory at sp-32 = 0x100-0x20 = 0xE0 should contain 0x78000
        let stored = tile.read_data_u32(0xE0);
        assert_eq!(stored, Some(0x78000), "st p7 must store p7's VALUE (0x78000), not 0");
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
        MemoryUnit::execute(&store_op, &mut ctx, &mut tile, None, None);

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
        MemoryUnit::execute(&load_op, &mut ctx, &mut tile, None, None);
        ctx.flush_pending_writes();

        assert_eq!(
            ctx.pointer.read(7),
            0x78000,
            "p7 must be restored to original value after store/load roundtrip"
        );
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

        MemoryUnit::execute(&op, &mut ctx, &mut tile, None, None);

        // sp-36 = 0x200-0x24 = 0x1DC
        let stored = tile.read_data_u32(0x1DC);
        assert_eq!(stored, Some(0x74000), "st p6 must store p6's value correctly");
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

        MemoryUnit::execute(&op, &mut ctx, &mut tile, None, None);

        let stored = tile.read_data_u32(0xFC); // 0x100 - 4
        assert_eq!(stored, Some(0x42), "st dj0 must store modifier register value");
    }

    /// Test the fused vlda.ups + vst.srs round-trip for s8.s32 wide (cm register).
    ///
    /// Reproduces the exact instruction sequence from the compiler-generated
    /// conversion test: vlda.ups.s32.s8 cm0, s0, [p0] -> vst.srs.s8.s32 cm0, s0, [p1].
    #[test]
    fn test_fused_ups_srs_s8_s32_wide_roundtrip() {
        use crate::interpreter::decode::register_map::AccumWidth;

        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Write 32 bytes of i8 test data to input address (0x100)
        let input_data: [u8; 32] = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
            27, 28, 29, 30, 31, 32,
        ];
        for (i, &byte) in input_data.iter().enumerate() {
            tile.data_memory_mut()[0x100 + i] = byte;
        }

        // Set up pointer registers: p0 = input (0x70100), p1 = output (0x70200)
        // Address 0x70100 decodes to local memory at offset 0x100
        ctx.pointer.write(0, 0x70100);
        ctx.pointer.write(1, 0x70200);

        // s0 = 0 (shift amount) -- matches what the compiler generates.
        // The sentinel pattern fills s0 with 0xDEADBEEF by default,
        // but we set it to 0 here to match hardware reset state.
        // A separate test below verifies sentinel behavior.
        ctx.scalar.write(0, 0);

        // Step 1: vlda.ups.s32.s8 cm0, s0, [p0, #0]
        let mut ups_op = SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Ups)
            .with_dest(Operand::AccumReg(0))
            .with_source(Operand::ScalarReg(0)) // shift register
            .with_source(Operand::Memory { base: 0, offset: 0 });
        ups_op.from_type = Some(ElementType::Int8);
        ups_op.element_type = Some(ElementType::Int32);
        ups_op.accum_width = Some(AccumWidth::Full); // cm register
        ups_op.mem_width = MemWidth::Vector256;
        ups_op.is_vector = true;

        MemoryUnit::execute(&ups_op, &mut ctx, &mut tile, None, None);

        // The UPS shift register is sampled at issue+6 (II_VLDA_UPS operand
        // cycle 7); advance the bundle clock and process the deferred load.
        ctx.bundle_seq += 6;
        ctx.process_pending_ups_loads();

        // Verify accumulator has non-zero data
        let acc = ctx.accumulator.read_wide(0);
        assert!(acc.iter().any(|&v| v != 0), "UPS should produce non-zero accumulator data");

        // Step 2: vst.srs.s8.s32 cm0, s0, [p1, #0]
        let mut srs_op = SlotOp::from_semantic(SlotIndex::Store, SemanticOp::Srs)
            .with_source(Operand::AccumReg(0)) // accumulator source
            .with_source(Operand::ScalarReg(0)) // shift register
            .with_source(Operand::Memory { base: 1, offset: 0 });
        srs_op.from_type = Some(ElementType::Int32);
        srs_op.element_type = Some(ElementType::Int8);
        srs_op.accum_width = Some(AccumWidth::Full); // cm register
        srs_op.mem_width = MemWidth::Vector256;
        srs_op.is_vector = true;

        MemoryUnit::execute(&srs_op, &mut ctx, &mut tile, None, None);

        // Read output and compare with input (should be identity for shift=0)
        let output: Vec<u8> = (0..32).map(|i| tile.data_memory()[0x200 + i]).collect();
        assert_eq!(
            &output[..],
            &input_data[..],
            "fused vlda.ups + vst.srs s8.s32 round-trip should be identity"
        );
    }

    /// Verify that s0 (SRS shift register) starts at zero, matching hardware
    /// reset state. The fused UPS/SRS pipeline depends on this because
    /// compilers may read s0 before writing it.
    #[test]
    fn test_fused_ups_srs_s0_zero_init() {
        use crate::interpreter::decode::register_map::AccumWidth;

        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // Write test data to input
        let input_data: [u8; 32] = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
            27, 28, 29, 30, 31, 32,
        ];
        for (i, &byte) in input_data.iter().enumerate() {
            tile.data_memory_mut()[0x100 + i] = byte;
        }

        ctx.pointer.write(0, 0x70100);
        ctx.pointer.write(1, 0x70200);

        // DO NOT set s0 explicitly. SRS shift register s0 = ScalarReg(40)
        // should already be zero from hardware-matching init.
        assert_eq!(ctx.scalar.read(40), 0, "s0 should start at 0 (hw reset)");

        // UPS with s0 = 0 (from init, like hardware)
        let mut ups_op = SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Ups)
            .with_dest(Operand::AccumReg(0))
            .with_source(Operand::ScalarReg(40)) // s0 = ScalarReg(40)
            .with_source(Operand::Memory { base: 0, offset: 0 });
        ups_op.from_type = Some(ElementType::Int8);
        ups_op.element_type = Some(ElementType::Int32);
        ups_op.accum_width = Some(AccumWidth::Full);
        ups_op.mem_width = MemWidth::Vector256;
        ups_op.is_vector = true;

        MemoryUnit::execute(&ups_op, &mut ctx, &mut tile, None, None);

        // Stage-7 shift sample: advance and process the deferred UPS load.
        ctx.bundle_seq += 6;
        ctx.process_pending_ups_loads();

        // SRS with s0 = 0 (unchanged)
        let mut srs_op = SlotOp::from_semantic(SlotIndex::Store, SemanticOp::Srs)
            .with_source(Operand::AccumReg(0))
            .with_source(Operand::ScalarReg(40)) // s0 = ScalarReg(40)
            .with_source(Operand::Memory { base: 1, offset: 0 });
        srs_op.from_type = Some(ElementType::Int32);
        srs_op.element_type = Some(ElementType::Int8);
        srs_op.accum_width = Some(AccumWidth::Full);
        srs_op.mem_width = MemWidth::Vector256;
        srs_op.is_vector = true;

        MemoryUnit::execute(&srs_op, &mut ctx, &mut tile, None, None);

        // s0=0 round-trip should produce identity
        let output: Vec<u8> = (0..32).map(|i| tile.data_memory()[0x200 + i]).collect();
        assert_eq!(&output[..], &input_data[..], "UPS->SRS with s0=0 (hw init) should round-trip correctly");
    }

    /// Fused `vlda.ups` with POST-INCREMENT addressing (`[p0], #32`) must route
    /// to the fused-load path: load from memory into the accumulator AND advance
    /// the pointer. Regression for the Half-B SRS kernel -- post-increment fused
    /// loads decode as PointerReg + post_modify with NO `Memory{}` operand, and
    /// `has_memory_operand` previously matched only `Memory{}`, so they fell
    /// through to the standalone register-UPS path: no memory load, no pointer
    /// advance, garbage accumulator (the real `vec_srs_i32` failure).
    #[test]
    fn test_fused_ups_post_increment_loads_and_advances_pointer() {
        use crate::interpreter::decode::register_map::AccumWidth;

        let mut ctx = make_ctx();
        let mut tile = make_tile();

        // 8 int32 at local 0x100.
        let vals: [i32; 8] = [10, 20, 30, 40, 50, 60, 70, 80];
        for (i, &v) in vals.iter().enumerate() {
            let b = (v as u32).to_le_bytes();
            for k in 0..4 {
                tile.data_memory_mut()[0x100 + i * 4 + k] = b[k];
            }
        }
        ctx.pointer.write(0, 0x70100); // p0 -> input
        ctx.scalar.write(41, 0); // s1 (UPS shift) = 0

        // vlda.ups.s64.s32 bml0, s1, [p0], #32  (post-increment, no Memory{} operand)
        let mut ups = SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Ups)
            .with_dest(Operand::AccumReg(0))
            .with_source(Operand::ScalarReg(41))
            .with_source(Operand::PointerReg(0));
        ups.from_type = Some(ElementType::Int32);
        ups.element_type = Some(ElementType::Int64);
        ups.accum_width = Some(AccumWidth::Half);
        ups.mem_width = MemWidth::Vector256;
        ups.is_vector = true;
        ups.post_modify = PostModify::Immediate(32);

        // The upstream dispatch runs VectorAlu BEFORE MemoryUnit. VectorAlu must
        // DEFER (return false) on a memory-addressing fused UPS so MemoryUnit
        // handles it; otherwise it consumes the op as a register-UPS (reading a
        // register as data, never touching memory) -- the real dispatch bug.
        assert!(
            !super::super::VectorAlu::execute(&ups, &mut ctx),
            "VectorAlu must skip a fused (memory-addressing) post-increment UPS"
        );

        let handled = MemoryUnit::execute(&ups, &mut ctx, &mut tile, None, None);
        assert!(handled, "post-increment vlda.ups must be handled by the fused-load path");

        // Stage-7 shift sample: advance and process the deferred UPS load.
        ctx.bundle_seq += 6;
        ctx.process_pending_ups_loads();

        // Accumulator holds the loaded int32 values (widened to acc64), not zeros.
        let acc = ctx.accumulator.read(0);
        assert_eq!(acc[0], 10, "bml0 lane0 must be the loaded value, not garbage");
        assert_eq!(acc[7], 80, "bml0 lane7 must be the loaded value");

        // p0 post-incremented by 32 bytes.
        assert_eq!(ctx.pointer.read(0), 0x70120, "p0 must post-increment by 32 bytes");
    }

    /// Fused `vst.srs` with POST-INCREMENT addressing (`[p1], #32`) decodes as
    /// sources = [acc, shift, PointerReg] + post_modify, with NO `Memory{}`
    /// operand. The store address must be the POINTER's value, not a positional
    /// `sources[1]` (which is the SHIFT register). Regression for the Half-B
    /// mode-sweep SRS kernels: a compile emitted the first `VST.SRS` as
    /// `[p1], #32`, and `get_store_address`'s positional fallback read the shift
    /// (=4) as the base -- scattering the first 16 outputs to address 0x4 while
    /// the indexed (`[p1, #off]` -> `Memory{}`) stores landed correctly. The
    /// fused-load path already searched for the pointer; the store path did not.
    #[test]
    fn test_fused_srs_post_increment_store_address_is_pointer() {
        ctx_and_op_assert_store_address(PostModify::Immediate(32), 0x70200);
    }

    /// The same store with a bare pointer and no post-modify (`[p1]`) must also
    /// address the pointer, not the shift register.
    #[test]
    fn test_fused_srs_bare_pointer_store_address_is_pointer() {
        ctx_and_op_assert_store_address(PostModify::None, 0x70200);
    }

    fn ctx_and_op_assert_store_address(post_modify: PostModify, expected: u32) {
        use crate::interpreter::decode::register_map::AccumWidth;
        let mut ctx = make_ctx();
        ctx.pointer.write(1, 0x70200);
        // shift = 4: the value that used to leak through as the store address.
        ctx.scalar.write(0, 4);
        let mut srs = SlotOp::from_semantic(SlotIndex::Store, SemanticOp::Srs)
            .with_source(Operand::AccumReg(0)) // acc data
            .with_source(Operand::ScalarReg(0)) // shift register (NOT the address)
            .with_source(Operand::PointerReg(1)); // address pointer
        srs.from_type = Some(ElementType::Int64);
        srs.element_type = Some(ElementType::Int16);
        srs.accum_width = Some(AccumWidth::Full);
        srs.mem_width = MemWidth::Vector256;
        srs.is_vector = true;
        srs.post_modify = post_modify;

        assert_eq!(
            MemoryUnit::get_store_address(&srs, &ctx),
            expected,
            "fused store must address the pointer, not the shift register"
        );
    }

    // --- Cross-tile load pipeline hazard regression test ---

    #[test]
    fn test_cross_tile_load_completes_at_same_latency_as_local() {
        // Regression test for cross-tile load latency bug.
        //
        // On AIE2, core loads from neighbor memory have the SAME pipeline
        // latency as local loads (7 cycles). The Chess compiler schedules
        // load-use distances accordingly. If the emulator adds extra latency
        // for cross-tile loads, stores scheduled at cycle+7 read stale
        // register values (the load hasn't written back yet).
        //
        // This test verifies: a cross-tile load into r0, committed after
        // exactly LATENCY_MEMORY cycles, produces the correct value.
        let mut device = crate::device::DeviceState::new_npu1();

        // Write a known value to the west neighbor's memory (tile 0,3)
        let test_value: u32 = 0xDEAD_BEEF;
        if let Some(west) = device.tile_mut(0, 3) {
            west.write_data_u32(0xA00, test_value);
        }

        // Set up execution context for tile (1,3)
        let mut ctx = ExecutionContext::new();
        let mut tile = Tile::compute(1, 3);

        // Pre-load r0 with a stale value (this is what the bug exposed:
        // the store would read this stale value instead of the loaded one)
        ctx.scalar.write(0, 0x00000003);

        // Point p0 at the west neighbor address: 0x50A00 (CardDir 5, offset 0xA00)
        ctx.pointer.write(0, 0x50A00);

        // Build neighbor snapshot
        let mut nbr = NeighborMemory::new(1, 3);
        nbr.ensure_snapshot(MemoryQuadrant::West, &device);

        // Execute: r0 = [p0] (load from west neighbor)
        let load_op = SlotOp::from_semantic(SlotIndex::LoadA, SemanticOp::Load)
            .with_dest(Operand::ScalarReg(0))
            .with_source(Operand::PointerReg(0));

        let issue_cycle = ctx.cycles;
        MemoryUnit::execute(&load_op, &mut ctx, &mut tile, Some(&mut nbr), None);

        // The load should NOT be visible yet (it's queued in the pipeline)
        assert_eq!(
            ctx.scalar.read(0),
            0x00000003,
            "Load should not be committed immediately (pipeline latency)"
        );

        // Advance to exactly LATENCY_MEMORY cycles after issue and commit.
        // This is the cycle where hardware makes the value available and
        // where the compiler schedules dependent instructions.
        ctx.cycles = issue_cycle + LATENCY_MEMORY as u64;
        ctx.commit_pending_writes();

        // The loaded value must be visible now -- at the SAME latency as
        // a local load. If cross-tile loads had extra latency, this would
        // still show the stale value (0x00000003).
        assert_eq!(
            ctx.scalar.read(0),
            test_value,
            "Cross-tile load must complete at LATENCY_MEMORY cycles, same as local"
        );
    }
}
