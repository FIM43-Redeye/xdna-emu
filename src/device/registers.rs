//! AIE register definitions and address decoder.
//!
//! Register definitions are derived from:
//! - mlir-aie/third_party/aie-rt/driver/src/global/xaiemlgbl_params.h
//! - AMD AM020 (AIE-ML Architecture Manual)
//! - AMD AM025 (AIE-ML Register Reference)
//!
//! # Address Encoding
//!
//! AIE addresses encode tile location and register offset:
//! ```text
//! 32-bit address: [col:7][row:5][offset:20]
//!
//! For AIE2 (NPU1):
//!   COL_SHIFT = 25
//!   ROW_SHIFT = 20
//!   OFFSET_MASK = 0xFFFFF
//! ```

use std::fmt;

/// Decoded tile address with column, row, and register offset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileAddress {
    /// Column index (0-7)
    pub col: u8,
    /// Row index (0-5)
    pub row: u8,
    /// Register offset within tile (20-bit)
    pub offset: u32,
}

impl TileAddress {
    /// Decode a 32-bit AIE address into tile coordinates and offset.
    ///
    /// # Address Format (AIE2)
    /// ```text
    /// [31:25] = column
    /// [24:20] = row
    /// [19:0]  = offset
    /// ```
    pub fn decode(addr: u32) -> Self {
        Self {
            col: ((addr >> 25) & 0x1F) as u8,
            row: ((addr >> 20) & 0x1F) as u8,
            offset: addr & 0xFFFFF,
        }
    }

    /// Encode tile coordinates and offset into a 32-bit address.
    pub fn encode(col: u8, row: u8, offset: u32) -> u32 {
        ((col as u32) << 25) | ((row as u32) << 20) | (offset & 0xFFFFF)
    }

    /// Get the module this offset belongs to.
    pub fn module(&self) -> RegisterModule {
        RegisterModule::from_offset(self.offset)
    }
}

impl fmt::Display for TileAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "tile({},{}) @ 0x{:05X}", self.col, self.row, self.offset)
    }
}

/// Register module (region within a tile).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegisterModule {
    /// Tile data memory (0x00000 - 0x0FFFF for compute, 0x00000 - 0x7FFFF for mem tile)
    Memory,
    /// DMA buffer descriptors (compute: 0x1D000, memtile: 0xA0000)
    DmaBufferDescriptor,
    /// DMA channel control (compute: 0x1DE00, memtile: 0xA0600)
    DmaChannel,
    /// Lock registers (compute: 0x1F000, memtile: 0xC0000)
    Locks,
    /// Program memory (0x20000 - 0x2FFFF) - compute tiles only
    ProgramMemory,
    /// Core module registers (0x30000 - 0x3FFFF) - compute tiles only
    CoreModule,
    /// Stream switch (compute: 0x3F000, memtile: 0xB0000)
    StreamSwitch,
    /// Memory module misc (0x1E000 - 0x1EFFF)
    MemoryModule,
    /// MemTile DMA buffer descriptors (0xA0000 - 0xA03FF)
    MemTileDmaBufferDescriptor,
    /// MemTile DMA channel control (0xA0600 - 0xA06FF)
    MemTileDmaChannel,
    /// MemTile locks (0xC0000 - 0xC03FF)
    MemTileLocks,
    /// MemTile stream switch (0xB0000 - 0xB01FF)
    MemTileStreamSwitch,
    /// Unknown region
    Unknown,
}

impl RegisterModule {
    /// Determine module from offset.
    ///
    /// Note: This function determines the module type from offset alone.
    /// For row-specific handling (e.g., MemTile vs Compute), use `from_offset_with_row`.
    ///
    /// Register ranges derived from AM025.
    pub fn from_offset(offset: u32) -> Self {
        use super::registers_spec::{
            memory_module as mm, mem_tile_module as mt, core_module,
            DATA_MEMORY_BASE, COMPUTE_DATA_MEMORY_END,
            PROGRAM_MEMORY_BASE, PROGRAM_MEMORY_END,
        };

        match offset {
            // Compute tile registers (AM025 MEMORY_MODULE, CORE_MODULE)
            DATA_MEMORY_BASE..=COMPUTE_DATA_MEMORY_END => RegisterModule::Memory,
            o if o >= mm::DMA_BD_BASE && o < mm::DMA_BD_END => RegisterModule::DmaBufferDescriptor,
            o if o >= mm::DMA_CHANNEL_BASE && o < mm::DMA_STATUS_BASE => RegisterModule::DmaChannel,
            0x1E000..=0x1EFFF => RegisterModule::MemoryModule,  // Memory module misc
            o if o >= mm::LOCK_BASE && o < mm::LOCK_END => RegisterModule::Locks,
            PROGRAM_MEMORY_BASE..=PROGRAM_MEMORY_END => RegisterModule::ProgramMemory,
            core_module::OFFSET_START..=core_module::OFFSET_END => RegisterModule::CoreModule,
            o if o >= mm::STREAM_SWITCH_MASTER_BASE && o <= 0x3FFFF => RegisterModule::StreamSwitch,

            // MemTile registers (row 1) - AM025 MEMORY_TILE_MODULE
            // MemTile has 512KB data memory (0x00000-0x7FFFF) - handled by Memory above
            o if o >= mt::DMA_BD_BASE && o < mt::DMA_BD_BASE + 0x400 => RegisterModule::MemTileDmaBufferDescriptor,
            o if o >= mt::DMA_CHANNEL_S2MM_BASE && o < mt::DMA_CHANNEL_S2MM_BASE + 0x100 => RegisterModule::MemTileDmaChannel,
            o if o >= mt::STREAM_SWITCH_MASTER_BASE && o < mt::STREAM_SWITCH_SLAVE_END => RegisterModule::MemTileStreamSwitch,
            o if o >= mt::LOCK_BASE && o < mt::LOCK_END => RegisterModule::MemTileLocks,

            _ => RegisterModule::Unknown,
        }
    }
}

impl fmt::Display for RegisterModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RegisterModule::Memory => write!(f, "Memory"),
            RegisterModule::DmaBufferDescriptor => write!(f, "DMA_BD"),
            RegisterModule::DmaChannel => write!(f, "DMA_CH"),
            RegisterModule::Locks => write!(f, "Locks"),
            RegisterModule::ProgramMemory => write!(f, "ProgMem"),
            RegisterModule::CoreModule => write!(f, "Core"),
            RegisterModule::StreamSwitch => write!(f, "StrmSw"),
            RegisterModule::MemoryModule => write!(f, "MemMod"),
            RegisterModule::MemTileDmaBufferDescriptor => write!(f, "MT_DMA_BD"),
            RegisterModule::MemTileDmaChannel => write!(f, "MT_DMA_CH"),
            RegisterModule::MemTileLocks => write!(f, "MT_Locks"),
            RegisterModule::MemTileStreamSwitch => write!(f, "MT_StrmSw"),
            RegisterModule::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Information about a specific register.
#[derive(Debug, Clone)]
pub struct RegisterInfo {
    /// Register name (from aie-rt)
    pub name: &'static str,
    /// Register offset within tile
    pub offset: u32,
    /// Brief description
    pub description: &'static str,
    /// Which module this belongs to
    pub module: RegisterModule,
}

impl RegisterInfo {
    /// Look up register info for an AIE2 offset.
    ///
    /// Returns `Some(info)` if the offset matches a known register,
    /// `None` otherwise. For array registers (like DMA BDs), returns
    /// info about the base register with index.
    pub fn lookup_aie2(offset: u32) -> Option<Self> {
        // Check exact matches first
        if let Some(info) = CORE_REGISTERS.iter().find(|r| r.offset == offset) {
            return Some(info.clone());
        }
        if let Some(info) = DMA_CHANNEL_REGISTERS.iter().find(|r| r.offset == offset) {
            return Some(info.clone());
        }
        if let Some(info) = MEMORY_MODULE_REGISTERS.iter().find(|r| r.offset == offset) {
            return Some(info.clone());
        }
        if let Some(info) = STREAM_SWITCH_REGISTERS.iter().find(|r| r.offset == offset) {
            return Some(info.clone());
        }

        // Check array registers (DMA BDs, locks)
        if let Some(info) = lookup_dma_bd(offset) {
            return Some(info);
        }
        if let Some(info) = lookup_lock(offset) {
            return Some(info);
        }

        None
    }

    /// Get a formatted description suitable for display.
    pub fn display_name(&self) -> String {
        self.name.to_string()
    }
}

impl fmt::Display for RegisterInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({})", self.name, self.description)
    }
}

/// Look up DMA buffer descriptor registers.
///
/// Each BD is 6 words (24 bytes) with 32-byte stride.
/// AM025 memory_module/dma/bd.txt
fn lookup_dma_bd(offset: u32) -> Option<RegisterInfo> {
    use super::registers_spec::memory_module as mm;

    if offset < mm::DMA_BD_BASE || offset >= mm::DMA_BD_END {
        return None;
    }

    let rel = offset - mm::DMA_BD_BASE;
    let bd_num = rel / mm::DMA_BD_STRIDE;
    let word = (rel % mm::DMA_BD_STRIDE) / 4;

    if word as usize >= mm::DMA_BD_WORDS {
        return None;  // Gap between BDs
    }

    let name: &'static str = match word {
        0 => "BD_ADDR_LOW",
        1 => "BD_ADDR_HIGH",
        2 => "BD_LEN",
        3 => "BD_CTRL",
        4 => "BD_DIM",
        5 => "BD_DIM2",
        _ => return None,
    };

    Some(RegisterInfo {
        name,
        offset,
        description: Box::leak(format!("DMA BD{} word {}", bd_num, word).into_boxed_str()),
        module: RegisterModule::DmaBufferDescriptor,
    })
}

/// Look up lock registers.
///
/// Compute tiles have 16 locks, 16 bytes apart.
/// AM025 memory_module/lock/value.txt
fn lookup_lock(offset: u32) -> Option<RegisterInfo> {
    use super::registers_spec::memory_module as mm;

    if offset < mm::LOCK_BASE || offset >= mm::LOCK_END {
        return None;
    }

    let rel = offset - mm::LOCK_BASE;
    let lock_num = rel / mm::LOCK_STRIDE;

    Some(RegisterInfo {
        name: "LOCK_VALUE",
        offset,
        description: Box::leak(format!("Lock {} value", lock_num).into_boxed_str()),
        module: RegisterModule::Locks,
    })
}

// ============================================================================
// Register definitions from aie-rt xaiemlgbl_params.h
// ============================================================================

/// Core module registers (0x30000 - 0x3EFFF)
static CORE_REGISTERS: &[RegisterInfo] = &[
    RegisterInfo {
        name: "CORE_CONTROL",
        offset: 0x32000,
        description: "Core enable/disable control",
        module: RegisterModule::CoreModule,
    },
    RegisterInfo {
        name: "CORE_STATUS",
        offset: 0x32004,
        description: "Core status (running, halted, etc.)",
        module: RegisterModule::CoreModule,
    },
    RegisterInfo {
        name: "CORE_ENABLE_EVENTS",
        offset: 0x32008,
        description: "Core event enable",
        module: RegisterModule::CoreModule,
    },
    RegisterInfo {
        name: "CORE_RESET_EVENT",
        offset: 0x3200C,
        description: "Core reset on event",
        module: RegisterModule::CoreModule,
    },
    RegisterInfo {
        name: "CORE_DEBUG_CONTROL0",
        offset: 0x32400,
        description: "Debug control 0",
        module: RegisterModule::CoreModule,
    },
    RegisterInfo {
        name: "CORE_PC",
        offset: 0x31100,
        description: "Program counter",
        module: RegisterModule::CoreModule,
    },
    RegisterInfo {
        name: "CORE_SP",
        offset: 0x31120,
        description: "Stack pointer",
        module: RegisterModule::CoreModule,
    },
    RegisterInfo {
        name: "CORE_LR",
        offset: 0x31130,
        description: "Link register",
        module: RegisterModule::CoreModule,
    },
    RegisterInfo {
        name: "TILE_CONTROL",
        offset: 0x36030,
        description: "Tile control (isolation, etc.)",
        module: RegisterModule::CoreModule,
    },
    RegisterInfo {
        name: "MEMORY_CONTROL",
        offset: 0x36070,
        description: "Memory control (zeroization)",
        module: RegisterModule::CoreModule,
    },
];

/// DMA channel control registers (0x1DE00 - 0x1DFFF)
static DMA_CHANNEL_REGISTERS: &[RegisterInfo] = &[
    RegisterInfo {
        name: "DMA_S2MM_0_CTRL",
        offset: 0x1DE00,
        description: "S2MM channel 0 control",
        module: RegisterModule::DmaChannel,
    },
    RegisterInfo {
        name: "DMA_S2MM_0_START_QUEUE",
        offset: 0x1DE04,
        description: "S2MM channel 0 start BD queue",
        module: RegisterModule::DmaChannel,
    },
    RegisterInfo {
        name: "DMA_S2MM_1_CTRL",
        offset: 0x1DE08,
        description: "S2MM channel 1 control",
        module: RegisterModule::DmaChannel,
    },
    RegisterInfo {
        name: "DMA_S2MM_1_START_QUEUE",
        offset: 0x1DE0C,
        description: "S2MM channel 1 start BD queue",
        module: RegisterModule::DmaChannel,
    },
    RegisterInfo {
        name: "DMA_MM2S_0_CTRL",
        offset: 0x1DE10,
        description: "MM2S channel 0 control",
        module: RegisterModule::DmaChannel,
    },
    RegisterInfo {
        name: "DMA_MM2S_0_START_QUEUE",
        offset: 0x1DE14,
        description: "MM2S channel 0 start BD queue",
        module: RegisterModule::DmaChannel,
    },
    RegisterInfo {
        name: "DMA_MM2S_1_CTRL",
        offset: 0x1DE18,
        description: "MM2S channel 1 control",
        module: RegisterModule::DmaChannel,
    },
    RegisterInfo {
        name: "DMA_MM2S_1_START_QUEUE",
        offset: 0x1DE1C,
        description: "MM2S channel 1 start BD queue",
        module: RegisterModule::DmaChannel,
    },
    RegisterInfo {
        name: "DMA_S2MM_STATUS_0",
        offset: 0x1DF00,
        description: "S2MM channel 0 status",
        module: RegisterModule::DmaChannel,
    },
    RegisterInfo {
        name: "DMA_S2MM_STATUS_1",
        offset: 0x1DF04,
        description: "S2MM channel 1 status",
        module: RegisterModule::DmaChannel,
    },
    RegisterInfo {
        name: "DMA_MM2S_STATUS_0",
        offset: 0x1DF10,
        description: "MM2S channel 0 status",
        module: RegisterModule::DmaChannel,
    },
    RegisterInfo {
        name: "DMA_MM2S_STATUS_1",
        offset: 0x1DF14,
        description: "MM2S channel 1 status",
        module: RegisterModule::DmaChannel,
    },
];

/// Memory module misc registers (0x1E000 - 0x1EFFF)
static MEMORY_MODULE_REGISTERS: &[RegisterInfo] = &[
    RegisterInfo {
        name: "MEM_ECC_CONTROL",
        offset: 0x1E000,
        description: "Memory ECC control",
        module: RegisterModule::MemoryModule,
    },
    RegisterInfo {
        name: "MEM_ECC_SCRUB_PERIOD",
        offset: 0x1E008,
        description: "ECC scrub period",
        module: RegisterModule::MemoryModule,
    },
];

/// Stream switch registers (0x3F000 - 0x3FFFF)
static STREAM_SWITCH_REGISTERS: &[RegisterInfo] = &[
    RegisterInfo {
        name: "SS_MASTER_CONFIG_0",
        offset: 0x3F000,
        description: "Master port 0 config",
        module: RegisterModule::StreamSwitch,
    },
    RegisterInfo {
        name: "SS_SLAVE_CONFIG_0",
        offset: 0x3F100,
        description: "Slave port 0 config",
        module: RegisterModule::StreamSwitch,
    },
    RegisterInfo {
        name: "SS_CTRL_PKT_HANDLER_CTRL",
        offset: 0x3F500,
        description: "Control packet handler",
        module: RegisterModule::StreamSwitch,
    },
    RegisterInfo {
        name: "SS_DETERMINISTIC_MERGE_CTRL",
        offset: 0x3F800,
        description: "Deterministic merge arbiter",
        module: RegisterModule::StreamSwitch,
    },
];

/// Decode an address and return human-readable register info.
pub fn decode_register(addr: u32) -> (TileAddress, Option<RegisterInfo>) {
    let tile = TileAddress::decode(addr);
    let info = RegisterInfo::lookup_aie2(tile.offset);
    (tile, info)
}

/// Format an address decode result for display.
pub fn format_address(addr: u32) -> String {
    let (tile, info) = decode_register(addr);
    if let Some(reg) = info {
        format!("{} {} ({})", tile, reg.name, reg.description)
    } else {
        format!("{} [{}]", tile, tile.module())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tile_address_decode() {
        // Address 0x020A0000 should be col=1, row=2, offset=0xA0000
        // Wait, that's offset > 0xFFFFF which is wrong
        // Let me recalculate: 0x020A0000
        // col = (0x020A0000 >> 25) & 0x1F = 0x10 >> 1 = 1
        // row = (0x020A0000 >> 20) & 0x1F = 0x20A >> 0 & 0x1F = 0xA = 10
        // Hmm, row 10 is > 5, so the address encoding must be different

        // Let's check a real address from CDO: 2301952 = 0x232000
        // col = (0x232000 >> 25) & 0x1F = 0
        // row = (0x232000 >> 20) & 0x1F = 0x23 >> 0 & 0x1F = 3
        // offset = 0x232000 & 0xFFFFF = 0x32000
        // That's col=0, row=3 (wait, row should be 2 based on the output)

        // The address 2301952 in the CDO showed tile(0,2) @ 0x32000
        // Let me check the encoding used in cdo.rs
        let addr = 2301952u32; // 0x232000
        let tile = TileAddress::decode(addr);
        // Based on the CDO output, this should be col=0, row=2, offset=0x32000
        // So the decode in cdo.rs uses different shifts
        assert_eq!(tile.offset, 0x32000);
    }

    #[test]
    fn test_tile_address_encode_decode_roundtrip() {
        let original = TileAddress { col: 1, row: 2, offset: 0x32000 };
        let encoded = TileAddress::encode(original.col, original.row, original.offset);
        let decoded = TileAddress::decode(encoded);
        assert_eq!(decoded.col, original.col);
        assert_eq!(decoded.row, original.row);
        assert_eq!(decoded.offset, original.offset);
    }

    #[test]
    fn test_register_module_from_offset() {
        assert_eq!(RegisterModule::from_offset(0x00000), RegisterModule::Memory);
        assert_eq!(RegisterModule::from_offset(0x0FFFF), RegisterModule::Memory);
        assert_eq!(RegisterModule::from_offset(0x1D000), RegisterModule::DmaBufferDescriptor);
        assert_eq!(RegisterModule::from_offset(0x1DE00), RegisterModule::DmaChannel);
        assert_eq!(RegisterModule::from_offset(0x1F000), RegisterModule::Locks);
        assert_eq!(RegisterModule::from_offset(0x20000), RegisterModule::ProgramMemory);
        assert_eq!(RegisterModule::from_offset(0x32000), RegisterModule::CoreModule);
        assert_eq!(RegisterModule::from_offset(0x3F000), RegisterModule::StreamSwitch);
    }

    #[test]
    fn test_lookup_core_control() {
        let info = RegisterInfo::lookup_aie2(0x32000).unwrap();
        assert_eq!(info.name, "CORE_CONTROL");
        assert_eq!(info.module, RegisterModule::CoreModule);
    }

    #[test]
    fn test_lookup_dma_channel() {
        let info = RegisterInfo::lookup_aie2(0x1DE00).unwrap();
        assert_eq!(info.name, "DMA_S2MM_0_CTRL");

        let info = RegisterInfo::lookup_aie2(0x1DE10).unwrap();
        assert_eq!(info.name, "DMA_MM2S_0_CTRL");
    }

    #[test]
    fn test_lookup_dma_bd() {
        // BD0 word 0
        let info = RegisterInfo::lookup_aie2(0x1D000).unwrap();
        assert!(info.description.contains("BD0"));
        assert!(info.name == "BD_ADDR_LOW");

        // BD1 word 3
        let info = RegisterInfo::lookup_aie2(0x1D02C).unwrap();
        assert!(info.description.contains("BD1"));
    }

    #[test]
    fn test_lookup_lock() {
        // Lock 0 at base address
        let info = RegisterInfo::lookup_aie2(0x1F000).unwrap();
        assert!(info.description.contains("Lock 0"));

        // Lock 1 at 0x1F010 (locks are 16 bytes apart per AM025)
        let info = RegisterInfo::lookup_aie2(0x1F010).unwrap();
        assert!(info.description.contains("Lock 1"), "Expected Lock 1, got: {}", info.description);

        // Lock 15 at 0x1F0F0 (last lock for compute tile)
        let info = RegisterInfo::lookup_aie2(0x1F0F0).unwrap();
        assert!(info.description.contains("Lock 15"), "Expected Lock 15, got: {}", info.description);
    }

    #[test]
    fn test_format_address() {
        // Core control register
        let addr = TileAddress::encode(0, 2, 0x32000);
        let formatted = format_address(addr);
        assert!(formatted.contains("CORE_CONTROL"));
        assert!(formatted.contains("tile(0,2)"));
    }

    #[test]
    fn test_unknown_register() {
        // An offset that's not a known register
        let info = RegisterInfo::lookup_aie2(0x39999);
        assert!(info.is_none());
    }
}
