//! Pre-resolved field layouts for hot-path register parsing.
//!
//! Each layout struct resolves string-based register/field lookups at startup
//! and stores the resulting `BitField` values for O(1) extraction during
//! emulation. This avoids repeated hash map lookups on every BD parse or
//! channel control access.

use crate::regdb::{BitField, RegisterDb};

// ============================================================================
// Compute tile BD layout (6 words)
// ============================================================================

/// Pre-resolved layout for compute tile BD registers (6 words).
///
/// Avoids repeated string lookups during BD parsing. Each field is a cloned
/// `BitField` with pre-computed mask/shift for O(1) extraction.
///
/// BD word layout (AM025 memory_module/dma/bd.txt):
/// - Word 0: Base_Address, Buffer_Length
/// - Word 1: Enable_Compression, Enable_Packet, Out_Of_Order_BD_ID, Packet_ID, Packet_Type
/// - Word 2: D0_Stepsize, D1_Stepsize
/// - Word 3: D0_Wrap, D1_Wrap, D2_Stepsize
/// - Word 4: Iteration_Current, Iteration_Wrap, Iteration_Stepsize
/// - Word 5: TLAST_Suppress, Next_BD, Use_Next_BD, Valid_BD, lock fields
#[derive(Debug, Clone)]
pub struct BdFieldLayout {
    // Word 0
    pub buffer_length: BitField,
    pub base_address: BitField,
    // Word 1
    pub enable_compression: BitField,
    pub enable_packet: BitField,
    pub out_of_order_bd_id: BitField,
    pub packet_id: BitField,
    pub packet_type: BitField,
    // Word 2
    pub d0_stepsize: BitField,
    pub d1_stepsize: BitField,
    // Word 3
    pub d0_wrap: BitField,
    pub d1_wrap: BitField,
    pub d2_stepsize: BitField,
    // Word 4
    pub iteration_current: BitField,
    pub iteration_wrap: BitField,
    pub iteration_stepsize: BitField,
    // Word 5
    pub tlast_suppress: BitField,
    pub next_bd: BitField,
    pub use_next_bd: BitField,
    pub valid_bd: BitField,
    pub lock_rel_value: BitField,
    pub lock_rel_id: BitField,
    pub lock_acq_enable: BitField,
    pub lock_acq_value: BitField,
    pub lock_acq_id: BitField,
}

impl BdFieldLayout {
    /// Build from the register database for a given module.
    ///
    /// Resolves all BD fields from DMA_BD0_0 through DMA_BD0_5 registers.
    pub fn from_regdb(db: &RegisterDb, module: &str) -> Result<Self, String> {
        let m = db.module(module)
            .ok_or_else(|| format!("Module '{}' not found in register database", module))?;

        // Helper to get a field from a register, with clear error messages
        let get_field = |reg_name: &str, field_name: &str| -> Result<BitField, String> {
            let reg = m.register(reg_name)
                .ok_or_else(|| format!("{}.{} not found", module, reg_name))?;
            reg.field(field_name)
                .cloned()
                .ok_or_else(|| format!("{}.{}.{} not found", module, reg_name, field_name))
        };

        Ok(Self {
            // Word 0: DMA_BD0_0
            buffer_length: get_field("DMA_BD0_0", "Buffer_Length")?,
            base_address: get_field("DMA_BD0_0", "Base_Address")?,
            // Word 1: DMA_BD0_1
            enable_compression: get_field("DMA_BD0_1", "Enable_Compression")?,
            enable_packet: get_field("DMA_BD0_1", "Enable_Packet")?,
            out_of_order_bd_id: get_field("DMA_BD0_1", "Out_Of_Order_BD_ID")?,
            packet_id: get_field("DMA_BD0_1", "Packet_ID")?,
            packet_type: get_field("DMA_BD0_1", "Packet_Type")?,
            // Word 2: DMA_BD0_2
            d0_stepsize: get_field("DMA_BD0_2", "D0_Stepsize")?,
            d1_stepsize: get_field("DMA_BD0_2", "D1_Stepsize")?,
            // Word 3: DMA_BD0_3
            d0_wrap: get_field("DMA_BD0_3", "D0_Wrap")?,
            d1_wrap: get_field("DMA_BD0_3", "D1_Wrap")?,
            d2_stepsize: get_field("DMA_BD0_3", "D2_Stepsize")?,
            // Word 4: DMA_BD0_4
            iteration_current: get_field("DMA_BD0_4", "Iteration_Current")?,
            iteration_wrap: get_field("DMA_BD0_4", "Iteration_Wrap")?,
            iteration_stepsize: get_field("DMA_BD0_4", "Iteration_Stepsize")?,
            // Word 5: DMA_BD0_5
            tlast_suppress: get_field("DMA_BD0_5", "TLAST_Suppress")?,
            next_bd: get_field("DMA_BD0_5", "Next_BD")?,
            use_next_bd: get_field("DMA_BD0_5", "Use_Next_BD")?,
            valid_bd: get_field("DMA_BD0_5", "Valid_BD")?,
            lock_rel_value: get_field("DMA_BD0_5", "Lock_Rel_Value")?,
            lock_rel_id: get_field("DMA_BD0_5", "Lock_Rel_ID")?,
            lock_acq_enable: get_field("DMA_BD0_5", "Lock_Acq_Enable")?,
            lock_acq_value: get_field("DMA_BD0_5", "Lock_Acq_Value")?,
            lock_acq_id: get_field("DMA_BD0_5", "Lock_Acq_ID")?,
        })
    }
}

// ============================================================================
// DMA channel control and status layouts
// ============================================================================

/// Pre-resolved layout for DMA channel control and start queue registers.
///
/// Covers both S2MM_Ctrl and MM2S_Ctrl field layouts (they share the same
/// bit assignments in AM025, differing only in which fields are meaningful).
#[derive(Debug, Clone)]
pub struct ChannelFieldLayout {
    // Control register fields
    pub fot_mode: BitField,
    pub controller_id: BitField,
    pub decompression_enable: BitField,
    pub enable_out_of_order: BitField,
    pub reset: BitField,
    // Start queue register fields
    pub enable_token_issue: BitField,
    pub repeat_count: BitField,
    pub start_bd_id: BitField,
}

impl ChannelFieldLayout {
    /// Build from the register database for a given module.
    ///
    /// Uses DMA_S2MM_0_Ctrl and DMA_S2MM_0_Start_Queue as the canonical
    /// source (all channels share the same field layout).
    pub fn from_regdb(db: &RegisterDb, module: &str) -> Result<Self, String> {
        let m = db.module(module)
            .ok_or_else(|| format!("Module '{}' not found in register database", module))?;

        let get_field = |reg_name: &str, field_name: &str| -> Result<BitField, String> {
            let reg = m.register(reg_name)
                .ok_or_else(|| format!("{}.{} not found", module, reg_name))?;
            reg.field(field_name)
                .cloned()
                .ok_or_else(|| format!("{}.{}.{} not found", module, reg_name, field_name))
        };

        Ok(Self {
            fot_mode: get_field("DMA_S2MM_0_Ctrl", "FoT_Mode")?,
            controller_id: get_field("DMA_S2MM_0_Ctrl", "Controller_ID")?,
            decompression_enable: get_field("DMA_S2MM_0_Ctrl", "Decompression_Enable")?,
            enable_out_of_order: get_field("DMA_S2MM_0_Ctrl", "Enable_Out_of_Order")?,
            reset: get_field("DMA_S2MM_0_Ctrl", "Reset")?,
            enable_token_issue: get_field("DMA_S2MM_0_Start_Queue", "Enable_Token_Issue")?,
            repeat_count: get_field("DMA_S2MM_0_Start_Queue", "Repeat_Count")?,
            start_bd_id: get_field("DMA_S2MM_0_Start_Queue", "Start_BD_ID")?,
        })
    }
}

/// Pre-resolved layout for DMA channel status registers.
///
/// Covers `DMA_S2MM_Status_0` / `DMA_MM2S_Status_0` fields. The same
/// field layout applies to all channels within a module (S2MM and MM2S
/// share the same bit assignments for status).
///
/// Key difference between modules: the `cur_bd` field is 4 bits [27:24]
/// for compute tiles (16 BDs) but 6 bits [29:24] for memtiles (48 BDs).
#[derive(Debug, Clone)]
pub struct StatusFieldLayout {
    pub status: BitField,               // [1:0] IDLE/STARTING/RUNNING
    pub stalled_lock_acq: BitField,     // [2]
    pub stalled_lock_rel: BitField,     // [3]
    pub stalled_stream: BitField,       // [4]
    pub stalled_tct: BitField,          // [5]
    pub error_bd_unavailable: BitField, // [10]
    pub error_bd_invalid: BitField,     // [11]
    pub task_queue_overflow: BitField,  // [18]
    pub channel_running: BitField,      // [19]
    pub task_queue_size: BitField,      // [22:20]
    pub cur_bd: BitField,               // [27:24] compute / [29:24] memtile
}

impl StatusFieldLayout {
    /// Build from the register database for a given module.
    ///
    /// Uses `DMA_S2MM_Status_0` as the canonical source (all channels
    /// share the same field layout within a module).
    pub fn from_regdb(db: &RegisterDb, module: &str) -> Result<Self, String> {
        let m = db.module(module)
            .ok_or_else(|| format!("Module '{}' not found in register database", module))?;

        let reg_name = "DMA_S2MM_Status_0";
        let reg = m.register(reg_name)
            .ok_or_else(|| format!("{}.{} not found", module, reg_name))?;

        let get_field = |field_name: &str| -> Result<BitField, String> {
            reg.field(field_name)
                .cloned()
                .ok_or_else(|| format!("{}.{}.{} not found", module, reg_name, field_name))
        };

        Ok(Self {
            status: get_field("Status")?,
            stalled_lock_acq: get_field("Stalled_Lock_Acq")?,
            stalled_lock_rel: get_field("Stalled_Lock_Rel")?,
            stalled_stream: get_field("Stalled_Stream_Starvation")?,
            stalled_tct: get_field("Stalled_TCT_or_Count_FIFO_Full")?,
            error_bd_unavailable: get_field("Error_BD_Unavailable")?,
            error_bd_invalid: get_field("Error_BD_Invalid")?,
            task_queue_overflow: get_field("Task_Queue_Overflow")?,
            channel_running: get_field("Channel_Running")?,
            task_queue_size: get_field("Task_Queue_Size")?,
            cur_bd: get_field("Cur_BD")?,
        })
    }
}

// ============================================================================
// MemTile BD layout (8 words)
// ============================================================================

/// Pre-resolved layout for MemTile BD registers (8 words).
///
/// MemTile BDs use 17-bit stepsizes and 19-bit addresses to cover the 512KB
/// address space. They also support 4-dimensional addressing (D3_Stepsize)
/// and zero-padding (D0/D1/D2 zero before/after) for MM2S transfers.
///
/// BD word layout (AM025 MEMORY_TILE_MODULE/DMA/BD):
/// - Word 0: Enable_Packet, Packet_Type, Packet_ID, OOO_BD_ID, Buffer_Length
/// - Word 1: D0_Zero_Before, Next_BD, Use_Next_BD, Base_Address
/// - Word 2: TLAST_Suppress, D0_Wrap, D0_Stepsize
/// - Word 3: D1_Zero_Before, D1_Wrap, D1_Stepsize
/// - Word 4: Enable_Compression, D2_Zero_Before, D2_Wrap, D2_Stepsize
/// - Word 5: D2_Zero_After, D1_Zero_After, D0_Zero_After, D3_Stepsize
/// - Word 6: Iteration_Current, Iteration_Wrap, Iteration_Stepsize
/// - Word 7: Valid_BD, Lock_Rel_Value, Lock_Rel_ID, Lock_Acq_Enable,
///           Lock_Acq_Value, Lock_Acq_ID
#[derive(Debug, Clone)]
pub struct MemTileBdFieldLayout {
    // Word 0
    pub enable_packet: BitField,
    pub packet_type: BitField,
    pub packet_id: BitField,
    pub out_of_order_bd_id: BitField,
    pub buffer_length: BitField,
    // Word 1
    pub d0_zero_before: BitField,
    pub next_bd: BitField,
    pub use_next_bd: BitField,
    pub base_address: BitField,
    // Word 2
    pub tlast_suppress: BitField,
    pub d0_wrap: BitField,
    pub d0_stepsize: BitField,
    // Word 3
    pub d1_zero_before: BitField,
    pub d1_wrap: BitField,
    pub d1_stepsize: BitField,
    // Word 4
    pub enable_compression: BitField,
    pub d2_zero_before: BitField,
    pub d2_wrap: BitField,
    pub d2_stepsize: BitField,
    // Word 5
    pub d2_zero_after: BitField,
    pub d1_zero_after: BitField,
    pub d0_zero_after: BitField,
    pub d3_stepsize: BitField,
    // Word 6
    pub iteration_current: BitField,
    pub iteration_wrap: BitField,
    pub iteration_stepsize: BitField,
    // Word 7
    pub valid_bd: BitField,
    pub lock_rel_value: BitField,
    pub lock_rel_id: BitField,
    pub lock_acq_enable: BitField,
    pub lock_acq_value: BitField,
    pub lock_acq_id: BitField,
}

impl MemTileBdFieldLayout {
    /// Build from the register database for the memory_tile module.
    ///
    /// Resolves all BD fields from DMA_BD0_0 through DMA_BD0_7 registers.
    pub fn from_regdb(db: &RegisterDb) -> Result<Self, String> {
        let module = "memory_tile";
        let m = db.module(module)
            .ok_or_else(|| format!("Module '{}' not found in register database", module))?;

        let get_field = |reg_name: &str, field_name: &str| -> Result<BitField, String> {
            let reg = m.register(reg_name)
                .ok_or_else(|| format!("{}.{} not found", module, reg_name))?;
            reg.field(field_name)
                .cloned()
                .ok_or_else(|| format!("{}.{}.{} not found", module, reg_name, field_name))
        };

        Ok(Self {
            // Word 0: DMA_BD0_0
            enable_packet: get_field("DMA_BD0_0", "Enable_Packet")?,
            packet_type: get_field("DMA_BD0_0", "Packet_Type")?,
            packet_id: get_field("DMA_BD0_0", "Packet_ID")?,
            out_of_order_bd_id: get_field("DMA_BD0_0", "Out_Of_Order_BD_ID")?,
            buffer_length: get_field("DMA_BD0_0", "Buffer_Length")?,
            // Word 1: DMA_BD0_1
            d0_zero_before: get_field("DMA_BD0_1", "D0_Zero_Before")?,
            next_bd: get_field("DMA_BD0_1", "Next_BD")?,
            use_next_bd: get_field("DMA_BD0_1", "Use_Next_BD")?,
            base_address: get_field("DMA_BD0_1", "Base_Address")?,
            // Word 2: DMA_BD0_2
            tlast_suppress: get_field("DMA_BD0_2", "TLAST_Suppress")?,
            d0_wrap: get_field("DMA_BD0_2", "D0_Wrap")?,
            d0_stepsize: get_field("DMA_BD0_2", "D0_Stepsize")?,
            // Word 3: DMA_BD0_3
            d1_zero_before: get_field("DMA_BD0_3", "D1_Zero_Before")?,
            d1_wrap: get_field("DMA_BD0_3", "D1_Wrap")?,
            d1_stepsize: get_field("DMA_BD0_3", "D1_Stepsize")?,
            // Word 4: DMA_BD0_4
            enable_compression: get_field("DMA_BD0_4", "Enable_Compression")?,
            d2_zero_before: get_field("DMA_BD0_4", "D2_Zero_Before")?,
            d2_wrap: get_field("DMA_BD0_4", "D2_Wrap")?,
            d2_stepsize: get_field("DMA_BD0_4", "D2_Stepsize")?,
            // Word 5: DMA_BD0_5
            d2_zero_after: get_field("DMA_BD0_5", "D2_Zero_After")?,
            d1_zero_after: get_field("DMA_BD0_5", "D1_Zero_After")?,
            d0_zero_after: get_field("DMA_BD0_5", "D0_Zero_After")?,
            d3_stepsize: get_field("DMA_BD0_5", "D3_Stepsize")?,
            // Word 6: DMA_BD0_6
            iteration_current: get_field("DMA_BD0_6", "Iteration_Current")?,
            iteration_wrap: get_field("DMA_BD0_6", "Iteration_Wrap")?,
            iteration_stepsize: get_field("DMA_BD0_6", "Iteration_Stepsize")?,
            // Word 7: DMA_BD0_7
            valid_bd: get_field("DMA_BD0_7", "Valid_BD")?,
            lock_rel_value: get_field("DMA_BD0_7", "Lock_Rel_Value")?,
            lock_rel_id: get_field("DMA_BD0_7", "Lock_Rel_ID")?,
            lock_acq_enable: get_field("DMA_BD0_7", "Lock_Acq_Enable")?,
            lock_acq_value: get_field("DMA_BD0_7", "Lock_Acq_Value")?,
            lock_acq_id: get_field("DMA_BD0_7", "Lock_Acq_ID")?,
        })
    }
}

// ============================================================================
// Shim BD layout (8 words)
// ============================================================================

/// Pre-resolved layout for Shim (NOC module) BD registers (8 words).
///
/// Shim BDs target DDR via NoC with AXI parameters not found in other tile types.
///
/// BD word layout (AM025 shim/dma/bd.txt):
/// - Word 0: Buffer_Length (full 32 bits for DDR transfers)
/// - Word 1: Base_Address_Low[31:2] (lower 30 bits of 46-bit word address)
/// - Word 2: Enable_Packet, OOO_ID, Packet_ID, Packet_Type, Base_Address_High[15:0]
/// - Word 3: Secure_Access, D0_Wrap, D0_Stepsize (20-bit)
/// - Word 4: Burst_Length, D1_Wrap, D1_Stepsize (20-bit)
/// - Word 5: SMID, AxCache, AxQoS, D2_Stepsize (20-bit)
/// - Word 6: Iteration_Current, Iteration_Wrap, Iteration_Stepsize (20-bit)
/// - Word 7: TLAST_Suppress, Next_BD, Use_Next_BD, Valid_BD, lock fields
#[derive(Debug, Clone)]
pub struct ShimBdFieldLayout {
    // Word 0
    pub buffer_length: BitField,
    // Word 1
    pub base_address_low: BitField,
    // Word 2
    pub enable_packet: BitField,
    pub out_of_order_bd_id: BitField,
    pub packet_id: BitField,
    pub packet_type: BitField,
    pub base_address_high: BitField,
    // Word 3
    pub secure_access: BitField,
    pub d0_wrap: BitField,
    pub d0_stepsize: BitField,
    // Word 4
    pub burst_length: BitField,
    pub d1_wrap: BitField,
    pub d1_stepsize: BitField,
    // Word 5
    pub smid: BitField,
    pub axcache: BitField,
    pub axqos: BitField,
    pub d2_stepsize: BitField,
    // Word 6
    pub iteration_current: BitField,
    pub iteration_wrap: BitField,
    pub iteration_stepsize: BitField,
    // Word 7
    pub tlast_suppress: BitField,
    pub next_bd: BitField,
    pub use_next_bd: BitField,
    pub valid_bd: BitField,
    pub lock_rel_value: BitField,
    pub lock_rel_id: BitField,
    pub lock_acq_enable: BitField,
    pub lock_acq_value: BitField,
    pub lock_acq_id: BitField,
}

impl ShimBdFieldLayout {
    /// Build from the register database for the shim module.
    ///
    /// Resolves all BD fields from DMA_BD0_0 through DMA_BD0_7 registers.
    pub fn from_regdb(db: &RegisterDb) -> Result<Self, String> {
        let module = "shim";
        let m = db.module(module)
            .ok_or_else(|| format!("Module '{}' not found in register database", module))?;

        let get_field = |reg_name: &str, field_name: &str| -> Result<BitField, String> {
            let reg = m.register(reg_name)
                .ok_or_else(|| format!("{}.{} not found", module, reg_name))?;
            reg.field(field_name)
                .cloned()
                .ok_or_else(|| format!("{}.{}.{} not found", module, reg_name, field_name))
        };

        Ok(Self {
            // Word 0: DMA_BD0_0
            buffer_length: get_field("DMA_BD0_0", "Buffer_Length")?,
            // Word 1: DMA_BD0_1
            base_address_low: get_field("DMA_BD0_1", "Base_Address_Low")?,
            // Word 2: DMA_BD0_2
            enable_packet: get_field("DMA_BD0_2", "Enable_Packet")?,
            out_of_order_bd_id: get_field("DMA_BD0_2", "Out_Of_Order_BD_ID")?,
            packet_id: get_field("DMA_BD0_2", "Packet_ID")?,
            packet_type: get_field("DMA_BD0_2", "Packet_Type")?,
            base_address_high: get_field("DMA_BD0_2", "Base_Address_High")?,
            // Word 3: DMA_BD0_3
            secure_access: get_field("DMA_BD0_3", "Secure_Access")?,
            d0_wrap: get_field("DMA_BD0_3", "D0_Wrap")?,
            d0_stepsize: get_field("DMA_BD0_3", "D0_Stepsize")?,
            // Word 4: DMA_BD0_4
            burst_length: get_field("DMA_BD0_4", "Burst_Length")?,
            d1_wrap: get_field("DMA_BD0_4", "D1_Wrap")?,
            d1_stepsize: get_field("DMA_BD0_4", "D1_Stepsize")?,
            // Word 5: DMA_BD0_5
            smid: get_field("DMA_BD0_5", "SMID")?,
            axcache: get_field("DMA_BD0_5", "AxCache")?,
            axqos: get_field("DMA_BD0_5", "AxQoS")?,
            d2_stepsize: get_field("DMA_BD0_5", "D2_Stepsize")?,
            // Word 6: DMA_BD0_6
            iteration_current: get_field("DMA_BD0_6", "Iteration_Current")?,
            iteration_wrap: get_field("DMA_BD0_6", "Iteration_Wrap")?,
            iteration_stepsize: get_field("DMA_BD0_6", "Iteration_Stepsize")?,
            // Word 7: DMA_BD0_7
            tlast_suppress: get_field("DMA_BD0_7", "TLAST_Suppress")?,
            next_bd: get_field("DMA_BD0_7", "Next_BD")?,
            use_next_bd: get_field("DMA_BD0_7", "Use_Next_BD")?,
            valid_bd: get_field("DMA_BD0_7", "Valid_BD")?,
            lock_rel_value: get_field("DMA_BD0_7", "Lock_Rel_Value")?,
            lock_rel_id: get_field("DMA_BD0_7", "Lock_Rel_ID")?,
            lock_acq_enable: get_field("DMA_BD0_7", "Lock_Acq_Enable")?,
            lock_acq_value: get_field("DMA_BD0_7", "Lock_Acq_Value")?,
            lock_acq_id: get_field("DMA_BD0_7", "Lock_Acq_ID")?,
        })
    }
}

// ============================================================================
// Shim mux/demux layout
// ============================================================================

/// A single shim mux/demux field mapping a register bit field to a
/// switchbox port index.
#[derive(Debug, Clone)]
pub struct ShimMuxField {
    /// Register bit field (2-bit select: 0=PL, 1=DMA, 2=NoC)
    pub field: BitField,
    /// Switchbox port index this field controls.
    /// For mux (input): slave port index.
    /// For demux (output): master port index.
    pub port_index: usize,
}

/// Pre-resolved layout for shim mux and demux configuration registers.
///
/// The shim mux selects which source (PL/DMA/NoC) feeds each South slave
/// port on the switchbox. The shim demux selects which destination receives
/// each South master port's output.
#[derive(Debug, Clone)]
pub struct ShimMuxLayout {
    /// Mux_Config register offset (e.g., 0x1F000)
    pub mux_offset: u32,
    /// Mux_Config fields: SouthN -> slave[N+2]
    pub mux_fields: Vec<ShimMuxField>,
    /// Demux_Config register offset (e.g., 0x1F004)
    pub demux_offset: u32,
    /// Demux_Config fields: SouthN -> master[N+2]
    pub demux_fields: Vec<ShimMuxField>,
}

impl ShimMuxLayout {
    /// Build from the register database for the shim module.
    ///
    /// Scans Mux_Config and Demux_Config register fields for "SouthN"
    /// names. Port index = N + 2 (South0=port[2] in shim switchbox layout).
    pub fn from_regdb(db: &RegisterDb) -> Result<Self, String> {
        let shim = db.module("shim")
            .ok_or("Module 'shim' not found in register database")?;

        let parse_south_fields = |reg_name: &str| -> Result<(u32, Vec<ShimMuxField>), String> {
            let reg = shim.register(reg_name)
                .ok_or_else(|| format!("shim.{} not found", reg_name))?;
            let mut fields = Vec::new();
            for bf in &reg.fields {
                if let Some(n_str) = bf.name.strip_prefix("South") {
                    if let Ok(n) = n_str.parse::<usize>() {
                        // SouthN maps to switchbox port N+2
                        // (South0=slave[2] / master[2] in shim layout)
                        fields.push(ShimMuxField {
                            field: bf.clone(),
                            port_index: n + 2,
                        });
                    }
                }
            }
            // Sort by port_index for deterministic ordering
            fields.sort_by_key(|f| f.port_index);
            Ok((reg.offset, fields))
        };

        let (mux_offset, mux_fields) = parse_south_fields("Mux_Config")?;
        let (demux_offset, demux_fields) = parse_south_fields("Demux_Config")?;

        Ok(Self {
            mux_offset,
            mux_fields,
            demux_offset,
            demux_fields,
        })
    }
}

// ============================================================================
// Stream switch layout
// ============================================================================

/// Pre-resolved stream switch register address ranges for one tile type.
///
/// Base addresses are derived from the first register in each group
/// (e.g., `Stream_Switch_Master_Config_AIE_Core0`). End addresses are
/// the offset of the last register plus one stride (exclusive end for
/// range checks). This replaces 6 hardcoded constants per module.
#[derive(Debug, Clone)]
pub struct StreamSwitchLayout {
    /// Stream switch master config base address
    pub master_base: u32,
    /// Stream switch master config end (exclusive)
    pub master_end: u32,
    /// Stream switch slave config base address
    pub slave_base: u32,
    /// Stream switch slave config end (exclusive)
    pub slave_end: u32,
    /// Stream switch slave slot config base address
    pub slave_slot_base: u32,
    /// Stream switch slave slot config end (exclusive)
    pub slave_slot_end: u32,
    /// Bytes between consecutive slave ports' slot registers (derived from AM025)
    pub slave_slot_port_stride: u32,
    /// Number of slots per slave port (derived from register count)
    pub slave_slot_count: usize,
}

impl StreamSwitchLayout {
    /// Build from the register database for a given module.
    ///
    /// Scans for the first and last `Stream_Switch_Master_Config_*`,
    /// `Stream_Switch_Slave_Config_*`, and `Stream_Switch_Slave_*_Slot*`
    /// registers to determine the exact address ranges.
    pub fn from_regdb(db: &RegisterDb, module: &str) -> Result<Self, String> {
        let m = db.module(module)
            .ok_or_else(|| format!("Module '{}' not found in register database", module))?;

        // Find all registers matching each group prefix
        let mut master_offsets = Vec::new();
        let mut slave_config_offsets = Vec::new();
        let mut slave_slot_offsets = Vec::new();

        for reg in &m.registers {
            if reg.name.starts_with("Stream_Switch_Master_Config_") {
                master_offsets.push(reg.offset);
            } else if reg.name.starts_with("Stream_Switch_Slave_Config_") {
                slave_config_offsets.push(reg.offset);
            } else if reg.name.contains("_Slot") && reg.name.starts_with("Stream_Switch_Slave_") {
                slave_slot_offsets.push(reg.offset);
            }
        }

        if master_offsets.is_empty() {
            return Err(format!("{}: no Stream_Switch_Master_Config registers found", module));
        }
        if slave_config_offsets.is_empty() {
            return Err(format!("{}: no Stream_Switch_Slave_Config registers found", module));
        }
        if slave_slot_offsets.is_empty() {
            return Err(format!("{}: no Stream_Switch_Slave slot registers found", module));
        }

        master_offsets.sort();
        slave_config_offsets.sort();
        slave_slot_offsets.sort();

        // Base = first register offset, End = last register offset + 4 (exclusive)
        let master_base = master_offsets[0];
        let master_end = master_offsets[master_offsets.len() - 1] + 4;
        let slave_base = slave_config_offsets[0];
        let slave_end = slave_config_offsets[slave_config_offsets.len() - 1] + 4;
        let slave_slot_base = slave_slot_offsets[0];
        let slave_slot_end = slave_slot_offsets[slave_slot_offsets.len() - 1] + 4;

        // Derive slave slot port stride and slot count from the register offsets.
        // Slot registers follow: Port0_Slot0, Port0_Slot1, ..., Port1_Slot0, ...
        // Within a port, slots are 4 bytes apart. Between ports, the stride is larger.
        // Find the first gap > 4 to determine where port 0 ends and port 1 begins.
        let num_slave_ports = slave_config_offsets.len();
        let (slave_slot_port_stride, slave_slot_count) = if num_slave_ports > 1
            && slave_slot_offsets.len() > 1
        {
            // Total slot registers / number of slave ports = slots per port
            let slots_per_port = slave_slot_offsets.len() / num_slave_ports;
            // Port stride = offset of port 1's first slot - offset of port 0's first slot
            let port_stride = if slots_per_port > 0 && slave_slot_offsets.len() > slots_per_port {
                slave_slot_offsets[slots_per_port] - slave_slot_offsets[0]
            } else {
                // Fallback: derive from gaps
                let mut stride = 0u32;
                for pair in slave_slot_offsets.windows(2) {
                    if pair[1] - pair[0] > 4 {
                        stride = pair[1] - pair[0] + 4 * (slots_per_port as u32 - 1);
                        break;
                    }
                }
                stride
            };
            (port_stride, slots_per_port)
        } else {
            (0x10, 4) // Shouldn't happen, but safe defaults
        };

        Ok(Self {
            master_base,
            master_end,
            slave_base,
            slave_end,
            slave_slot_base,
            slave_slot_end,
            slave_slot_port_stride,
            slave_slot_count,
        })
    }
}

// ============================================================================
// Per-module event/trace register layout
// ============================================================================

/// Per-module event and trace register layout.
///
/// Looked up by register name from the AM025 database, so offsets adapt
/// automatically for different architectures (AIE2, AIE2P, etc.).
#[derive(Debug, Clone)]
pub struct ModuleEventLayout {
    /// Trace_Control0 offset (first trace config register)
    pub trace_control_base: u32,
    /// Trace_Event1 offset (last trace config register, inclusive)
    pub trace_control_end: u32,
    /// Event_Generate offset
    pub event_generate: u32,
    /// Event_Broadcast0 offset (first broadcast channel register)
    pub event_broadcast_base: u32,
    /// Last broadcast channel register offset (inclusive)
    pub event_broadcast_end: u32,
    /// Edge_Detection_event_control offset
    pub edge_detection: u32,
    /// Stream_Switch_Event_Port_Selection_0 and _1 offsets
    pub event_port_select: Option<[u32; 2]>,
}
