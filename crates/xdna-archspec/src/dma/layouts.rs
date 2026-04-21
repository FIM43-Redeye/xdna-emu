//! Pre-resolved register layouts for one device architecture.
//!
//! Subsystem 3 migrated this aggregator from xdna-emu's `src/device/regdb/`
//! into archspec. Archspec owns the layout data (BD field offsets, BD
//! stride, channel base addresses, etc.); xdna-emu owns the runtime
//! cache (`OnceLock`) and config-system wiring (`load_for_device()`).
//!
//! Lock-value-width fields (`lock_value_mask` / `lock_value_sign_bit`)
//! and `sign_extend_lock_value` deliberately stayed in xdna-emu for
//! Subsystem 3 and migrate to `LockModel` as part of Subsystem 4; this
//! avoids a half-migrated lock-width concept straddling the crate
//! boundary during the refactor.

use crate::dma::field_layouts::{
    BdFieldLayout, ChannelFieldLayout, StatusFieldLayout,
    MemTileBdFieldLayout, ShimBdFieldLayout, ShimMuxLayout,
    StreamSwitchLayout, ModuleEventLayout,
};
use crate::regdb::RegisterDb;

/// Pre-resolved register layouts for one device architecture.
///
/// This aggregates all the field layouts needed by the emulator's hot paths
/// (BD parsing, channel control, lock access) into a single struct that is
/// resolved once at startup.
///
/// Structural constants (base addresses, strides) are derived from register
/// offsets in the JSON database, eliminating the need for hand-coded constants.
#[derive(Debug, Clone)]
pub struct DeviceRegLayout {
    /// Full register database (for ad-hoc lookups)
    pub db: RegisterDb,
    /// Compute tile BD field layout
    pub memory_bd: BdFieldLayout,
    /// DMA channel field layout (compute tiles)
    pub memory_channel: ChannelFieldLayout,
    /// DMA channel status field layout (compute tiles)
    pub memory_status: StatusFieldLayout,
    /// MemTile BD field layout
    pub memtile_bd: MemTileBdFieldLayout,
    /// MemTile DMA channel field layout
    pub memtile_channel: ChannelFieldLayout,
    /// MemTile DMA channel status field layout
    pub memtile_status: StatusFieldLayout,

    // -- Compute tile lock layout (derived from Lock0_value, Lock1_value) --
    /// Lock register base offset (memory module)
    pub memory_lock_base: u32,
    /// Lock register stride (memory module)
    pub memory_lock_stride: u32,
    /// Lock overflow status register (memory module, write-to-clear)
    pub memory_locks_overflow: u32,
    /// Lock underflow status register (memory module, write-to-clear)
    pub memory_locks_underflow: u32,

    // -- Compute tile BD layout (derived from DMA_BD0_0, DMA_BD1_0) --
    /// DMA BD base address (memory module)
    pub memory_bd_base: u32,
    /// DMA BD stride in bytes (memory module)
    pub memory_bd_stride: u32,
    /// Words per BD (number of DMA_BD0_N registers found)
    pub memory_bd_words: usize,

    // -- Compute tile channel layout (derived from DMA_S2MM_0_Ctrl etc.) --
    /// DMA channel control base address (memory module)
    pub memory_channel_base: u32,
    /// DMA channel stride in bytes (memory module)
    pub memory_channel_stride: u32,
    /// DMA channel status base address (memory module)
    pub memory_status_base: u32,

    // -- MemTile lock layout (derived from Lock0_value, Lock1_value) --
    /// Lock register base offset (memory tile)
    pub memtile_lock_base: u32,
    /// Lock register stride (memory tile)
    pub memtile_lock_stride: u32,
    /// Lock overflow status register 0 (memory tile, locks 0-31)
    pub memtile_locks_overflow_0: u32,
    /// Lock overflow status register 1 (memory tile, locks 32-63)
    pub memtile_locks_overflow_1: u32,
    /// Lock underflow status register 0 (memory tile, locks 0-31)
    pub memtile_locks_underflow_0: u32,
    /// Lock underflow status register 1 (memory tile, locks 32-63)
    pub memtile_locks_underflow_1: u32,

    // -- MemTile BD layout (derived from DMA_BD0_0, DMA_BD1_0) --
    /// DMA BD base address (memory tile)
    pub memtile_bd_base: u32,
    /// DMA BD stride in bytes (memory tile)
    pub memtile_bd_stride: u32,
    /// Words per BD (number of DMA_BD0_N registers found)
    pub memtile_bd_words: usize,

    // -- MemTile channel layout (derived from DMA_S2MM_0_Ctrl etc.) --
    /// S2MM channel control base address (memory tile)
    pub memtile_channel_s2mm_base: u32,
    /// MM2S channel control base address (memory tile)
    pub memtile_channel_mm2s_base: u32,
    /// DMA channel stride in bytes (memory tile)
    pub memtile_channel_stride: u32,

    // -- Stream switch layouts --
    /// Compute tile stream switch address ranges
    pub memory_stream_switch: StreamSwitchLayout,
    /// MemTile stream switch address ranges
    pub memtile_stream_switch: StreamSwitchLayout,

    // -- Shim mux/demux layout --
    /// Shim mux/demux register field layout
    pub shim_mux: ShimMuxLayout,

    // -- Shim lock layout (derived from Lock0_value, Lock1_value in shim module) --
    /// Lock register base offset (shim)
    pub shim_lock_base: u32,
    /// Lock register stride (shim)
    pub shim_lock_stride: u32,
    /// Lock overflow status register (shim, write-to-clear)
    pub shim_locks_overflow: u32,
    /// Lock underflow status register (shim, write-to-clear)
    pub shim_locks_underflow: u32,

    // -- Shim BD field layout --
    /// Shim (NOC module) BD field layout
    pub shim_bd: ShimBdFieldLayout,

    // -- Shim BD structural layout (derived from DMA_BD0_0, DMA_BD1_0) --
    /// DMA BD base address (shim)
    pub shim_bd_base: u32,
    /// DMA BD stride in bytes (shim)
    pub shim_bd_stride: u32,
    /// Words per BD (number of DMA_BD0_N registers found)
    pub shim_bd_words: usize,

    // -- Shim channel layout (derived from DMA_S2MM_0_Ctrl etc.) --
    /// DMA channel control base address (shim)
    pub shim_channel_base: u32,
    /// DMA channel stride in bytes (shim)
    pub shim_channel_stride: u32,

    // -- Per-module event/trace register layout (derived from register names) --
    // These are looked up by name from the register database, so they
    // automatically adapt when switching to a different architecture
    // (e.g., AIE2P with different register offsets).
    /// Core module event registers (compute + shim tiles)
    pub core_events: ModuleEventLayout,
    /// Memory module event registers (compute tiles)
    pub memory_events: ModuleEventLayout,
    /// MemTile event registers
    pub memtile_events: ModuleEventLayout,

    /// Cascade configuration register (compute tiles, core module)
    pub cascade_config_offset: u32,
}

impl DeviceRegLayout {
    /// Build from a register database, resolving all field layouts.
    ///
    /// Derives structural constants (base addresses, strides) from register
    /// offsets. For example, the BD base is DMA_BD0_0's offset, and the BD
    /// stride is DMA_BD1_0 - DMA_BD0_0. This eliminates hand-coded constants.
    pub fn from_regdb(db: RegisterDb) -> Result<Self, String> {
        let memory_bd = BdFieldLayout::from_regdb(&db, "memory")?;
        let memory_channel = ChannelFieldLayout::from_regdb(&db, "memory")?;
        let memory_status = StatusFieldLayout::from_regdb(&db, "memory")?;
        let memtile_bd = MemTileBdFieldLayout::from_regdb(&db)?;
        let memtile_channel = ChannelFieldLayout::from_regdb(&db, "memory_tile")?;
        let memtile_status = StatusFieldLayout::from_regdb(&db, "memory_tile")?;

        // Helper: get a register offset from a module
        let reg_offset = |module: &str, reg: &str| -> Result<u32, String> {
            db.module(module)
                .ok_or_else(|| format!("Module '{}' not found", module))?
                .register(reg)
                .ok_or_else(|| format!("{}.{} not found", module, reg))
                .map(|r| r.offset)
        };

        // -- Compute tile locks --
        let memory_lock_base = reg_offset("memory", "Lock0_value")?;
        let memory_lock_stride = reg_offset("memory", "Lock1_value")? - memory_lock_base;
        let memory_locks_overflow = reg_offset("memory", "Locks_Overflow")?;
        let memory_locks_underflow = reg_offset("memory", "Locks_Underflow")?;

        // -- Compute tile BDs --
        let memory_bd_base = reg_offset("memory", "DMA_BD0_0")?;
        let memory_bd_stride = reg_offset("memory", "DMA_BD1_0")? - memory_bd_base;

        // Count BD words: DMA_BD0_0 through DMA_BD0_N (contiguous registers)
        let mem = db.module("memory").unwrap();
        let memory_bd_words = (0..16)
            .take_while(|i| mem.register(&format!("DMA_BD0_{}", i)).is_some())
            .count();

        // -- Compute tile channels --
        let memory_channel_base = reg_offset("memory", "DMA_S2MM_0_Ctrl")?;
        let memory_channel_stride = reg_offset("memory", "DMA_S2MM_1_Ctrl")? - memory_channel_base;
        let memory_status_base = reg_offset("memory", "DMA_S2MM_Status_0")?;

        // -- MemTile locks --
        let memtile_lock_base = reg_offset("memory_tile", "Lock0_value")?;
        let memtile_lock_stride = reg_offset("memory_tile", "Lock1_value")? - memtile_lock_base;
        let memtile_locks_overflow_0 = reg_offset("memory_tile", "Locks_Overflow_0")?;
        let memtile_locks_overflow_1 = reg_offset("memory_tile", "Locks_Overflow_1")?;
        let memtile_locks_underflow_0 = reg_offset("memory_tile", "Locks_Underflow_0")?;
        let memtile_locks_underflow_1 = reg_offset("memory_tile", "Locks_Underflow_1")?;

        // -- MemTile BDs --
        let memtile_bd_base = reg_offset("memory_tile", "DMA_BD0_0")?;
        let memtile_bd_stride = reg_offset("memory_tile", "DMA_BD1_0")? - memtile_bd_base;

        let mt = db.module("memory_tile").unwrap();
        let memtile_bd_words = (0..16)
            .take_while(|i| mt.register(&format!("DMA_BD0_{}", i)).is_some())
            .count();

        // -- MemTile channels --
        let memtile_channel_s2mm_base = reg_offset("memory_tile", "DMA_S2MM_0_Ctrl")?;
        let memtile_channel_mm2s_base = reg_offset("memory_tile", "DMA_MM2S_0_Ctrl")?;
        let memtile_channel_stride = reg_offset("memory_tile", "DMA_S2MM_1_Ctrl")? - memtile_channel_s2mm_base;

        // -- Stream switch --
        // Note: AM025 JSON places compute tile stream switch registers under
        // "core" module (not "memory"), even though their addresses (0x3F000+)
        // are beyond the core module's general range (0x30000-0x3EFFF).
        let memory_stream_switch = StreamSwitchLayout::from_regdb(&db, "core")?;
        let memtile_stream_switch = StreamSwitchLayout::from_regdb(&db, "memory_tile")?;

        // -- Shim mux/demux --
        let shim_mux = ShimMuxLayout::from_regdb(&db)?;

        // -- Shim locks --
        let shim_lock_base = reg_offset("shim", "Lock0_value")?;
        let shim_lock_stride = reg_offset("shim", "Lock1_value")? - shim_lock_base;
        let shim_locks_overflow = reg_offset("shim", "Locks_Overflow")?;
        let shim_locks_underflow = reg_offset("shim", "Locks_Underflow")?;

        // -- Shim BD fields --
        let shim_bd = ShimBdFieldLayout::from_regdb(&db)?;

        // -- Shim BDs --
        let shim_bd_base = reg_offset("shim", "DMA_BD0_0")?;
        let shim_bd_stride = reg_offset("shim", "DMA_BD1_0")? - shim_bd_base;

        let sh = db.module("shim").unwrap();
        let shim_bd_words = (0..16)
            .take_while(|i| sh.register(&format!("DMA_BD0_{}", i)).is_some())
            .count();

        // -- Shim channels --
        let shim_channel_base = reg_offset("shim", "DMA_S2MM_0_Ctrl")?;
        let shim_channel_stride = reg_offset("shim", "DMA_S2MM_1_Ctrl")? - shim_channel_base;

        // -- Per-module event/trace registers --
        // Helper: look up a register offset, returning 0 if not found
        // (some modules may not have all event registers).
        let reg_offset_opt = |module: &str, reg: &str| -> u32 {
            db.module(module)
                .and_then(|m| m.register(reg))
                .map(|r| r.offset)
                .unwrap_or(0)
        };

        let build_event_layout = |module: &str| -> ModuleEventLayout {
            let trace_control_base = reg_offset_opt(module, "Trace_Control0");
            let trace_control_end = reg_offset_opt(module, "Trace_Event1");
            let event_generate = reg_offset_opt(module, "Event_Generate");
            // Broadcast register naming: "Event_Broadcast0" in core/memory/
            // memory_tile, but "Event_Broadcast0_A" in shim. Try both.
            let mut event_broadcast_base = reg_offset_opt(module, "Event_Broadcast0");
            if event_broadcast_base == 0 {
                event_broadcast_base = reg_offset_opt(module, "Event_Broadcast0_A");
            }
            // 16 broadcast channels at stride 4: last = base + 15*4
            let event_broadcast_end = if event_broadcast_base != 0 {
                event_broadcast_base + 15 * 4
            } else {
                0
            };
            let edge_detection = reg_offset_opt(module, "Edge_Detection_event_control");
            let port_sel_0 = reg_offset_opt(module, "Stream_Switch_Event_Port_Selection_0");
            let port_sel_1 = reg_offset_opt(module, "Stream_Switch_Event_Port_Selection_1");
            let event_port_select = if port_sel_0 != 0 {
                Some([port_sel_0, port_sel_1])
            } else {
                None
            };
            ModuleEventLayout {
                trace_control_base,
                trace_control_end,
                event_generate,
                event_broadcast_base,
                event_broadcast_end,
                edge_detection,
                event_port_select,
            }
        };

        let core_events = build_event_layout("core");
        let memory_events = build_event_layout("memory");
        let memtile_events = build_event_layout("memory_tile");

        // Cascade/accumulator control register (core module, compute tiles).
        // AM025 names this "Accumulator_Control" -- it configures cascade
        // input/output directions (bit 0 = input dir, bit 1 = output dir).
        let cascade_config_offset = reg_offset_opt("core", "Accumulator_Control");

        Ok(Self {
            memory_bd,
            memory_channel,
            memory_status,
            memtile_bd,
            memtile_channel,
            memtile_status,
            memory_lock_base,
            memory_lock_stride,
            memory_locks_overflow,
            memory_locks_underflow,
            memory_bd_base,
            memory_bd_stride,
            memory_bd_words,
            memory_channel_base,
            memory_channel_stride,
            memory_status_base,
            memtile_lock_base,
            memtile_lock_stride,
            memtile_locks_overflow_0,
            memtile_locks_overflow_1,
            memtile_locks_underflow_0,
            memtile_locks_underflow_1,
            memtile_bd_base,
            memtile_bd_stride,
            memtile_bd_words,
            memtile_channel_s2mm_base,
            memtile_channel_mm2s_base,
            memtile_channel_stride,
            memory_stream_switch,
            memtile_stream_switch,
            shim_mux,
            shim_lock_base,
            shim_lock_stride,
            shim_locks_overflow,
            shim_locks_underflow,
            shim_bd,
            shim_bd_base,
            shim_bd_stride,
            shim_bd_words,
            shim_channel_base,
            shim_channel_stride,
            core_events,
            memory_events,
            memtile_events,
            cascade_config_offset,
            db,
        })
    }
}
