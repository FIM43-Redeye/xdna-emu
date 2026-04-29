// AIE2 event subsystem emulation.
//
// The event system is how hardware subsystems communicate within a tile.
// Each tile module (core, memory, memtile, shim/PL) has its own event
// module with:
//
//   - Event status registers: 128-bit (4 x 32-bit) bit-packed status
//     tracking which events are currently asserted.
//   - Event generate register: software-triggered event injection.
//   - Group events: OR of related sub-events into a single output.
//   - Combo events: boolean logic (AND/OR/NOT) on pairs of input events.
//   - Broadcast channels: 16 channels to propagate events to neighbors.
//   - Port events: monitor stream switch port activity.
//   - Edge detection: rising/falling edge detection on events.
//
// This module is derived from the aie-rt event implementation:
//   - Event IDs: xaie_events_aieml.h (XAIEML_EVENTS_*)
//   - Register layout: xaiemlgbl_params.h (*_EVENT_*)
//   - Module config: xaiemlgbl_reginit.c (AieMlTileEvntMod, etc.)
//   - API semantics: xaie_events.c (XAie_Event*)
//
// Key design points from aie-rt:
//   - Core/memory modules: 128 events (7-bit IDs, 0-127)
//   - MemTile module: 161 events (8-bit IDs, 0-160)
//   - PL module: 128 events (7-bit IDs, 0-127)
//   - Event status: 4 registers (core/mem/PL) or 6 registers (memtile)
//   - Default status register 0 value: 0x2 (TRUE event always asserted)
//   - All modules have 16 broadcast channels
//   - Memory module of AIE tiles has NO port events

pub mod broadcast;
pub mod combo;
pub mod group;
pub mod port;

use broadcast::{BroadcastConfig, BroadcastDir};
use combo::{ComboEventConfig, ComboLogic, EventId};
use group::GroupEventConfig;
use port::PortEventConfig;

/// Module type for event configuration.
///
/// Determines the event ID space, group event layout, and available
/// features for this event module instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventModuleType {
    /// Core module of an AIE tile (128 events, 9 groups, port events).
    Core,
    /// Memory module of an AIE tile (128 events, 8 groups, NO port events).
    Memory,
    /// PL/NoC (shim) module (128 events, 6 groups, port events).
    Pl,
    /// MemTile module (161 events, 9 groups, port events).
    MemTile,
}

impl EventModuleType {
    /// Number of event status registers for this module type.
    ///
    /// Per aie-rt: NumEventReg = 4U for core/mem/PL, 6U for memtile.
    pub fn num_status_registers(self) -> usize {
        match self {
            EventModuleType::MemTile => 6,
            _ => 4,
        }
    }

    /// Total number of events in this module's event space.
    ///
    /// Per aie-rt: event IDs 0-127 for core/mem/PL, 0-160 for memtile.
    pub fn num_events(self) -> usize {
        match self {
            EventModuleType::MemTile => 161,
            _ => 128,
        }
    }

    /// Event ID bit width (7 for core/mem/PL, 8 for memtile).
    pub fn event_id_width(self) -> u8 {
        match self {
            EventModuleType::MemTile => 8,
            _ => 7,
        }
    }

    /// Event ID mask for register fields.
    pub fn event_id_mask(self) -> u8 {
        match self {
            EventModuleType::MemTile => 0xFF,
            _ => 0x7F,
        }
    }

    /// Whether this module has stream switch port events.
    pub fn has_port_events(self) -> bool {
        match self {
            EventModuleType::Memory => false,
            _ => true,
        }
    }

    /// Combo event base ID in this module's event space.
    ///
    /// Per aie-rt: COMBO_EVENT_0 IDs from xaie_events_aieml.h.
    pub fn combo_event_base(self) -> EventId {
        match self {
            EventModuleType::Core => 9,    // XAIEML_EVENTS_CORE_COMBO_EVENT_0
            EventModuleType::Memory => 7,  // XAIEML_EVENTS_MEM_COMBO_EVENT_0
            EventModuleType::Pl => 7,      // XAIEML_EVENTS_PL_COMBO_EVENT_0
            EventModuleType::MemTile => 9, // XAIEML_EVENTS_MEM_TILE_COMBO_EVENT_0
        }
    }

    /// Broadcast event base ID in this module's event space.
    ///
    /// Per aie-rt xaie_events_aieml.h.
    pub fn broadcast_event_base(self) -> EventId {
        match self {
            EventModuleType::Core => 107,    // XAIEML_EVENTS_CORE_BROADCAST_0
            EventModuleType::Memory => 107,  // XAIEML_EVENTS_MEM_BROADCAST_0
            EventModuleType::Pl => 110,      // XAIEML_EVENTS_PL_BROADCAST_A_0
            EventModuleType::MemTile => 142, // XAIEML_EVENTS_MEM_TILE_BROADCAST_0
        }
    }

    /// User event base ID in this module's event space.
    ///
    /// Per aie-rt xaie_events_aieml.h.
    pub fn user_event_base(self) -> EventId {
        match self {
            EventModuleType::Core => 124,    // XAIEML_EVENTS_CORE_USER_EVENT_0
            EventModuleType::Memory => 124,  // XAIEML_EVENTS_MEM_USER_EVENT_0
            EventModuleType::Pl => 126,      // XAIEML_EVENTS_PL_USER_EVENT_0
            EventModuleType::MemTile => 159, // XAIEML_EVENTS_MEM_TILE_USER_EVENT_0
        }
    }

    /// Number of user events.
    ///
    /// Per aie-rt: 4 for core/mem, 2 for PL/memtile.
    pub fn num_user_events(self) -> usize {
        match self {
            EventModuleType::Core => 4,
            EventModuleType::Memory => 4,
            EventModuleType::Pl => 2,
            EventModuleType::MemTile => 2,
        }
    }

    /// Port idle event base ID (for modules with port events).
    ///
    /// Per aie-rt PortIdleEventBase from xaiemlgbl_reginit.c.
    pub fn port_idle_event_base(self) -> Option<EventId> {
        match self {
            EventModuleType::Core => Some(74),    // XAIEML_EVENTS_CORE_PORT_IDLE_0
            EventModuleType::Memory => None,      // No port events
            EventModuleType::Pl => Some(77),      // XAIEML_EVENTS_PL_PORT_IDLE_0
            EventModuleType::MemTile => Some(79), // XAIEML_EVENTS_MEM_TILE_PORT_IDLE_0
        }
    }
}

/// Complete event module for one tile module.
///
/// Combines event status tracking, group events, combo events, broadcast
/// channels, and port events into a single coherent module that mirrors
/// the hardware event subsystem.
///
/// Event status is tracked as bit-packed 32-bit registers. The TRUE event
/// (ID 1) is always asserted (status register 0 default = 0x2).
#[derive(Debug, Clone)]
pub struct EventModule {
    /// Module type determines event counts and feature availability.
    module_type: EventModuleType,

    /// Event status registers (bit-packed, 32 events per register).
    ///
    /// Per aie-rt: 4 registers for core/mem/PL, 6 for memtile.
    /// Default value for register 0 is 0x2 (TRUE event always set).
    event_status: Vec<u32>,

    /// Group event configuration.
    pub group_events: GroupEventConfig,

    /// Combo event configuration (4 combo events per module).
    pub combo_events: ComboEventConfig,

    /// Broadcast channel configuration (16 channels per module).
    pub broadcast: BroadcastConfig,

    /// Port event configuration (8 slots, unavailable for memory module).
    pub port_events: PortEventConfig,

    /// Pending events waiting to be consumed by trace unit / perf counters.
    /// Each entry is a hardware event ID that fired since last drain.
    pending: Vec<EventId>,
}

impl EventModule {
    /// Create a new event module for the given module type.
    ///
    /// Initializes all sub-modules with hardware reset values:
    /// - Event status register 0 = 0x2 (TRUE event always asserted)
    /// - Group events get their default enable masks
    /// - Combo events, broadcast, port events start in reset state
    pub fn new(module_type: EventModuleType) -> Self {
        let num_regs = module_type.num_status_registers();
        let mut event_status = vec![0u32; num_regs];
        // TRUE event (ID 1) is always asserted. Per aie-rt:
        // EVENT_STATUS0 default value = 0x2.
        if !event_status.is_empty() {
            event_status[0] = 0x2;
        }

        let group_events = match module_type {
            EventModuleType::Core => GroupEventConfig::core(),
            EventModuleType::Memory => GroupEventConfig::memory(),
            EventModuleType::Pl => GroupEventConfig::pl(),
            EventModuleType::MemTile => GroupEventConfig::mem_tile(),
        };

        Self {
            module_type,
            event_status,
            group_events,
            combo_events: ComboEventConfig::new(),
            broadcast: BroadcastConfig::new(),
            port_events: PortEventConfig::new(module_type.has_port_events()),
            pending: Vec::new(),
        }
    }

    /// Get the module type.
    pub fn module_type(&self) -> EventModuleType {
        self.module_type
    }

    // -- Event generation and status --

    /// Generate (assert) an event by hardware event ID.
    ///
    /// Sets the corresponding bit in the event status registers and
    /// adds the event to the pending queue for consumers (trace unit,
    /// performance counters).
    ///
    /// Per aie-rt XAie_EventGenerate(): writes the event ID to the
    /// Event_Generate register, which pulses the event wire.
    pub fn generate_event(&mut self, event_id: EventId) {
        let max_events = self.module_type.num_events();
        if (event_id as usize) >= max_events {
            return;
        }

        let reg_idx = event_id as usize / 32;
        let bit_pos = event_id as usize % 32;

        if reg_idx < self.event_status.len() {
            self.event_status[reg_idx] |= 1 << bit_pos;
        }

        self.pending.push(event_id);
    }

    /// Clear (deassert) an event by hardware event ID.
    ///
    /// Clears the corresponding bit in the event status registers.
    /// The TRUE event (ID 1) cannot be cleared.
    pub fn clear_event(&mut self, event_id: EventId) {
        // TRUE event is always asserted.
        if event_id == 1 {
            return;
        }

        let reg_idx = event_id as usize / 32;
        let bit_pos = event_id as usize % 32;

        if reg_idx < self.event_status.len() {
            self.event_status[reg_idx] &= !(1 << bit_pos);
        }
    }

    /// Check if an event is currently asserted.
    pub fn is_event_active(&self, event_id: EventId) -> bool {
        let reg_idx = event_id as usize / 32;
        let bit_pos = event_id as usize % 32;

        if reg_idx < self.event_status.len() {
            (self.event_status[reg_idx] & (1 << bit_pos)) != 0
        } else {
            false
        }
    }

    /// Read an event status register by index (0-3 or 0-5 for memtile).
    pub fn read_event_status(&self, reg_index: usize) -> u32 {
        if reg_index < self.event_status.len() {
            self.event_status[reg_index]
        } else {
            0
        }
    }

    /// Drain and return all pending events since last drain.
    ///
    /// Consumers (trace unit, performance counters) call this each cycle
    /// to get the events that fired.
    pub fn drain_pending(&mut self) -> Vec<EventId> {
        std::mem::take(&mut self.pending)
    }

    /// Peek at pending events without consuming them.
    pub fn pending_events(&self) -> &[EventId] {
        &self.pending
    }

    // -- Group event interface --

    /// Configure a group event's enable mask.
    ///
    /// `group_id` is the index into this module's group event array (not
    /// the hardware event ID). Use `group_events.find_by_event_id()` to
    /// map from hardware event ID to group index.
    pub fn configure_group(&mut self, group_id: usize, mask: u32) {
        if group_id < self.group_events.count() {
            self.group_events.groups[group_id].configure(mask);
        }
    }

    // -- Combo event interface --

    /// Configure a combo event.
    ///
    /// For combo_id 2 (meta-combo), inputs are ignored -- it always
    /// combines the outputs of combo 0 and combo 1.
    pub fn configure_combo(
        &mut self,
        combo_id: usize,
        input_a: EventId,
        input_b: EventId,
        logic: ComboLogic,
    ) {
        self.combo_events.configure(combo_id, input_a, input_b, logic);
    }

    /// Evaluate combo events and generate any that fire.
    ///
    /// Should be called after other events are generated to allow
    /// combo events to react to the current event state.
    pub fn evaluate_combos(&mut self) {
        let base = self.module_type.combo_event_base();
        let fired = self.combo_events.evaluate(&|id| self.is_event_active(id), base);
        for event_id in fired {
            self.generate_event(event_id);
        }
    }

    // -- Broadcast interface --

    /// Configure a broadcast channel to broadcast the given event.
    pub fn configure_broadcast(&mut self, channel: usize, event: EventId) {
        self.broadcast.configure_channel(channel, event);
    }

    /// Set broadcast blocking for a channel in a direction.
    pub fn set_broadcast_block(&mut self, channel: usize, dir: BroadcastDir, blocked: bool) {
        if blocked {
            self.broadcast.block_channel(channel, dir);
        } else {
            self.broadcast.unblock_channel(channel, dir);
        }
    }

    // -- Port event interface --

    /// Configure a port event slot.
    pub fn configure_port_event(&mut self, slot: usize, port_id: u8, is_master: bool) {
        self.port_events.configure_slot(slot, port_id, is_master);
    }

    // -- Register interface --
    //
    // Register offsets are module-relative. The caller (tile.rs) is
    // responsible for translating absolute tile addresses to module-
    // relative offsets before calling these methods.
    //
    // Core module register map (from xaiemlgbl_params.h):
    //   0x4008: Event_Generate
    //   0x4010-0x404C: Event_Broadcast0-15 (stride 4)
    //   0x4050: Broadcast_Block_South_Set
    //   0x4054: Broadcast_Block_South_Clr
    //   0x4058: Broadcast_Block_South_Value
    //   0x4060: Broadcast_Block_West_Set
    //   ... (stride 0x10 per direction)
    //   0x4200-0x420C: Event_Status0-3
    //   0x4400: Combo_Event_Inputs
    //   0x4404: Combo_Event_Control
    //   0x4408: Edge_Detection_Event_Control
    //   0x4500-0x4520: Event_Group_*_Enable (stride 4)
    //
    // These offsets are relative to the module base (0x30000 for core,
    // 0x10000 for memory, etc.). The offsets below use the low 16 bits.

    /// Read a register at the given module-relative offset.
    ///
    /// Returns Some(value) if the offset maps to a known event register,
    /// None if unrecognized.
    pub fn read_register(&self, offset: u32) -> Option<u32> {
        // Mask to module-local offset (low 16 bits).
        let local = offset & 0xFFFF;

        match local {
            // Event_Generate: write-only in hardware, but we return 0.
            0x4008 => Some(0),

            // Event_Broadcast0-15: 0x4010 + channel * 4
            off @ 0x4010..=0x404C if (off - 0x4010) % 4 == 0 => {
                let channel = ((off - 0x4010) / 4) as usize;
                Some(self.broadcast.read_channel(channel))
            }

            // Broadcast block VALUE registers (read current state).
            // South=0x4058, West=0x4068, North=0x4078, East=0x4088
            0x4058 => Some(self.broadcast.read_block_value(BroadcastDir::South) as u32),
            0x4068 => Some(self.broadcast.read_block_value(BroadcastDir::West) as u32),
            0x4078 => Some(self.broadcast.read_block_value(BroadcastDir::North) as u32),
            0x4088 => Some(self.broadcast.read_block_value(BroadcastDir::East) as u32),

            // Event_Status0-3 (or 0-5 for memtile): 0x4200 + reg * 4
            off @ 0x4200..=0x4214 if (off - 0x4200) % 4 == 0 => {
                let reg = ((off - 0x4200) / 4) as usize;
                Some(self.read_event_status(reg))
            }

            // Combo_Event_Inputs
            0x4400 => Some(self.combo_events.read_input_register()),

            // Combo_Event_Control
            0x4404 => Some(self.combo_events.read_control_register()),

            // Event_Group_*_Enable: 0x4500 + group * 4
            off @ 0x4500..=0x4520 if (off - 0x4500) % 4 == 0 => {
                let group = ((off - 0x4500) / 4) as usize;
                if group < self.group_events.count() {
                    Some(self.group_events.groups[group].read_register())
                } else {
                    Some(0)
                }
            }

            // Stream_Switch_Event_Port_Selection_0/1
            0x4FF0 => Some(self.port_events.read_register(0)),
            0x4FF4 => Some(self.port_events.read_register(1)),

            _ => None,
        }
    }

    /// Write a register at the given module-relative offset.
    ///
    /// Returns true if the offset mapped to a known event register.
    pub fn write_register(&mut self, offset: u32, value: u32) -> bool {
        let local = offset & 0xFFFF;

        match local {
            // Event_Generate: triggers the event with the given ID.
            0x4008 => {
                let event_id = (value & self.module_type.event_id_mask() as u32) as u8;
                self.generate_event(event_id);
                true
            }

            // Event_Broadcast0-15: 0x4010 + channel * 4
            off @ 0x4010..=0x404C if (off - 0x4010) % 4 == 0 => {
                let channel = ((off - 0x4010) / 4) as usize;
                let event = (value & self.module_type.event_id_mask() as u32) as u8;
                self.broadcast.configure_channel(channel, event);
                true
            }

            // Broadcast block SET registers.
            // South=0x4050, West=0x4060, North=0x4070, East=0x4080
            0x4050 => {
                self.broadcast.write_block_set(BroadcastDir::South, value as u16);
                true
            }
            0x4060 => {
                self.broadcast.write_block_set(BroadcastDir::West, value as u16);
                true
            }
            0x4070 => {
                self.broadcast.write_block_set(BroadcastDir::North, value as u16);
                true
            }
            0x4080 => {
                self.broadcast.write_block_set(BroadcastDir::East, value as u16);
                true
            }

            // Broadcast block CLR registers.
            // South=0x4054, West=0x4064, North=0x4074, East=0x4084
            0x4054 => {
                self.broadcast.write_block_clr(BroadcastDir::South, value as u16);
                true
            }
            0x4064 => {
                self.broadcast.write_block_clr(BroadcastDir::West, value as u16);
                true
            }
            0x4074 => {
                self.broadcast.write_block_clr(BroadcastDir::North, value as u16);
                true
            }
            0x4084 => {
                self.broadcast.write_block_clr(BroadcastDir::East, value as u16);
                true
            }

            // Combo_Event_Inputs
            0x4400 => {
                self.combo_events.write_input_register(value);
                true
            }

            // Combo_Event_Control
            0x4404 => {
                self.combo_events.write_control_register(value);
                true
            }

            // Event_Group_*_Enable: 0x4500 + group * 4
            off @ 0x4500..=0x4520 if (off - 0x4500) % 4 == 0 => {
                let group = ((off - 0x4500) / 4) as usize;
                if group < self.group_events.count() {
                    self.group_events.groups[group].write_register(value);
                }
                true
            }

            // Stream_Switch_Event_Port_Selection_0/1
            0x4FF0 => {
                self.port_events.write_register(0, value);
                true
            }
            0x4FF4 => {
                self.port_events.write_register(1, value);
                true
            }

            _ => false,
        }
    }

    /// Reset the entire event module to hardware defaults.
    pub fn reset(&mut self) {
        // Reset status registers.
        for reg in &mut self.event_status {
            *reg = 0;
        }
        if !self.event_status.is_empty() {
            self.event_status[0] = 0x2; // TRUE event
        }

        // Reset group events to default masks.
        for group in &mut self.group_events.groups {
            group.reset();
        }

        self.combo_events.reset();
        self.broadcast.reset();
        self.port_events.reset();
        self.pending.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Construction tests --

    #[test]
    fn test_new_core() {
        let em = EventModule::new(EventModuleType::Core);
        assert_eq!(em.module_type(), EventModuleType::Core);
        assert_eq!(em.event_status.len(), 4);
        // TRUE event is always set.
        assert_eq!(em.read_event_status(0), 0x2);
        assert_eq!(em.read_event_status(1), 0);
        assert_eq!(em.group_events.count(), 9);
        assert!(em.port_events.available);
    }

    #[test]
    fn test_new_memory() {
        let em = EventModule::new(EventModuleType::Memory);
        assert_eq!(em.event_status.len(), 4);
        assert_eq!(em.group_events.count(), 8);
        assert!(!em.port_events.available);
    }

    #[test]
    fn test_new_pl() {
        let em = EventModule::new(EventModuleType::Pl);
        assert_eq!(em.event_status.len(), 4);
        assert_eq!(em.group_events.count(), 6);
        assert!(em.port_events.available);
    }

    #[test]
    fn test_new_memtile() {
        let em = EventModule::new(EventModuleType::MemTile);
        assert_eq!(em.event_status.len(), 6);
        assert_eq!(em.group_events.count(), 9);
        assert!(em.port_events.available);
    }

    // -- Event generation and status tests --

    #[test]
    fn test_generate_event_sets_status_bit() {
        let mut em = EventModule::new(EventModuleType::Core);
        em.generate_event(5);
        assert!(em.is_event_active(5));
        // Check the raw register.
        assert_eq!(em.read_event_status(0) & (1 << 5), 1 << 5);
    }

    #[test]
    fn test_generate_event_adds_to_pending() {
        let mut em = EventModule::new(EventModuleType::Core);
        em.generate_event(10);
        em.generate_event(20);
        let pending = em.pending_events();
        assert_eq!(pending.len(), 2);
        assert!(pending.contains(&10));
        assert!(pending.contains(&20));
    }

    #[test]
    fn test_clear_event_removes_status_bit() {
        let mut em = EventModule::new(EventModuleType::Core);
        em.generate_event(5);
        assert!(em.is_event_active(5));
        em.clear_event(5);
        assert!(!em.is_event_active(5));
    }

    #[test]
    fn test_clear_true_event_noop() {
        let mut em = EventModule::new(EventModuleType::Core);
        // TRUE event (ID 1) should always remain set.
        assert!(em.is_event_active(1));
        em.clear_event(1);
        assert!(em.is_event_active(1));
    }

    #[test]
    fn test_drain_pending() {
        let mut em = EventModule::new(EventModuleType::Core);
        em.generate_event(10);
        em.generate_event(20);
        let drained = em.drain_pending();
        assert_eq!(drained.len(), 2);
        // After drain, pending is empty.
        assert!(em.pending_events().is_empty());
    }

    #[test]
    fn test_event_in_high_register() {
        let mut em = EventModule::new(EventModuleType::Core);
        // Event 64 should be in status register 2 (64/32=2), bit 0.
        em.generate_event(64);
        assert!(em.is_event_active(64));
        assert_eq!(em.read_event_status(2) & 1, 1);
    }

    #[test]
    fn test_event_127_core() {
        let mut em = EventModule::new(EventModuleType::Core);
        em.generate_event(127);
        assert!(em.is_event_active(127));
        // Register 3 (127/32=3), bit 31 (127%32=31).
        assert_eq!(em.read_event_status(3) & (1 << 31), 1 << 31);
    }

    #[test]
    fn test_event_out_of_range_ignored() {
        let mut em = EventModule::new(EventModuleType::Core);
        // Core has 128 events (0-127). Event 128 should be ignored.
        em.generate_event(128);
        assert!(!em.is_event_active(128));
        assert!(em.pending_events().is_empty());
    }

    #[test]
    fn test_memtile_high_event() {
        let mut em = EventModule::new(EventModuleType::MemTile);
        // MemTile has 161 events (0-160).
        em.generate_event(160);
        assert!(em.is_event_active(160));
        // Register 5 (160/32=5), bit 0 (160%32=0).
        assert_eq!(em.read_event_status(5) & 1, 1);
    }

    #[test]
    fn test_multiple_events_simultaneously() {
        let mut em = EventModule::new(EventModuleType::Core);
        em.generate_event(5);
        em.generate_event(10);
        em.generate_event(33);
        assert!(em.is_event_active(5));
        assert!(em.is_event_active(10));
        assert!(em.is_event_active(33));
        assert!(!em.is_event_active(6));
    }

    #[test]
    fn test_event_status_register_read_out_of_bounds() {
        let em = EventModule::new(EventModuleType::Core);
        assert_eq!(em.read_event_status(10), 0);
    }

    // -- Group event tests --

    #[test]
    fn test_configure_group() {
        let mut em = EventModule::new(EventModuleType::Core);
        em.configure_group(0, 0x0003);
        assert_eq!(em.group_events.groups[0].enable_mask, 0x0003);
    }

    #[test]
    fn test_configure_group_out_of_bounds() {
        let mut em = EventModule::new(EventModuleType::Core);
        // Should be silently ignored.
        em.configure_group(100, 0x0003);
    }

    // -- Combo event tests --

    #[test]
    fn test_configure_combo() {
        let mut em = EventModule::new(EventModuleType::Core);
        em.configure_combo(0, 5, 10, ComboLogic::And);
        assert_eq!(em.combo_events.combos[0].input_a, 5);
        assert_eq!(em.combo_events.combos[0].input_b, 10);
        assert_eq!(em.combo_events.combos[0].logic, ComboLogic::And);
    }

    #[test]
    fn test_evaluate_combos_fires_event() {
        let mut em = EventModule::new(EventModuleType::Core);
        // Combo 0: event 5 OR event 10.
        em.configure_combo(0, 5, 10, ComboLogic::Or);
        // Generate event 5.
        em.generate_event(5);
        // Evaluate combos -- should fire COMBO_EVENT_0 (ID 9 for core).
        em.evaluate_combos();
        assert!(em.is_event_active(9));
    }

    #[test]
    fn test_evaluate_combos_and_not_satisfied() {
        let mut em = EventModule::new(EventModuleType::Core);
        // Combo 0: event 5 AND event 10.
        em.configure_combo(0, 5, 10, ComboLogic::And);
        // Only generate event 5 (not 10).
        em.generate_event(5);
        em.evaluate_combos();
        // COMBO_EVENT_0 should NOT fire.
        assert!(!em.is_event_active(9));
    }

    // -- Broadcast tests --

    #[test]
    fn test_configure_broadcast() {
        let mut em = EventModule::new(EventModuleType::Core);
        em.configure_broadcast(0, 42);
        assert_eq!(em.broadcast.channels[0].event, 42);
    }

    #[test]
    fn test_set_broadcast_block() {
        let mut em = EventModule::new(EventModuleType::Core);
        em.set_broadcast_block(0, BroadcastDir::South, true);
        assert!(em.broadcast.is_blocked(0, BroadcastDir::South));
        em.set_broadcast_block(0, BroadcastDir::South, false);
        assert!(!em.broadcast.is_blocked(0, BroadcastDir::South));
    }

    #[test]
    fn test_broadcast_forwards_event() {
        let mut em = EventModule::new(EventModuleType::Core);
        em.configure_broadcast(0, 42);
        em.generate_event(42);

        let pending = em.broadcast.pending_broadcasts(&|id| em.is_event_active(id));
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0], (0, 42));
    }

    #[test]
    fn test_broadcast_blocking_prevents_direction() {
        let mut em = EventModule::new(EventModuleType::Core);
        em.configure_broadcast(0, 42);
        em.set_broadcast_block(0, BroadcastDir::South, true);
        em.set_broadcast_block(0, BroadcastDir::East, true);

        let dirs = em.broadcast.allowed_directions(0);
        assert_eq!(dirs.len(), 2);
        assert!(dirs.contains(&BroadcastDir::West));
        assert!(dirs.contains(&BroadcastDir::North));
    }

    // -- Port event tests --

    #[test]
    fn test_configure_port_event() {
        let mut em = EventModule::new(EventModuleType::Core);
        em.configure_port_event(0, 5, true);
        assert_eq!(em.port_events.slots[0].port_id, 5);
        assert!(em.port_events.slots[0].is_master);
    }

    #[test]
    fn test_configure_port_event_memory_module() {
        let mut em = EventModule::new(EventModuleType::Memory);
        // Memory module has no port events -- should be ignored.
        em.configure_port_event(0, 5, true);
        assert_eq!(em.port_events.slots[0].port_id, 0);
    }

    // -- Register interface tests --

    #[test]
    fn test_register_event_generate() {
        let mut em = EventModule::new(EventModuleType::Core);
        // Write event ID 42 to Event_Generate register.
        assert!(em.write_register(0x4008, 42));
        assert!(em.is_event_active(42));
    }

    #[test]
    fn test_register_event_status_read() {
        let mut em = EventModule::new(EventModuleType::Core);
        em.generate_event(5);
        let val = em.read_register(0x4200);
        assert_eq!(val, Some(em.read_event_status(0)));
    }

    #[test]
    fn test_register_broadcast_rw() {
        let mut em = EventModule::new(EventModuleType::Core);
        // Write event 42 to broadcast channel 0.
        assert!(em.write_register(0x4010, 42));
        assert_eq!(em.read_register(0x4010), Some(42));
        // Channel 1.
        assert!(em.write_register(0x4014, 55));
        assert_eq!(em.read_register(0x4014), Some(55));
    }

    #[test]
    fn test_register_broadcast_block_set_clr() {
        let mut em = EventModule::new(EventModuleType::Core);
        // Set south block for channels 0 and 2.
        assert!(em.write_register(0x4050, 0x0005));
        assert_eq!(em.read_register(0x4058), Some(0x0005)); // VALUE register
                                                            // Clear channel 0 block.
        assert!(em.write_register(0x4054, 0x0001));
        assert_eq!(em.read_register(0x4058), Some(0x0004));
    }

    #[test]
    fn test_register_combo_input_rw() {
        let mut em = EventModule::new(EventModuleType::Core);
        assert!(em.write_register(0x4400, 0x12345678));
        assert_eq!(em.read_register(0x4400), Some(0x12345678));
    }

    #[test]
    fn test_register_combo_control_rw() {
        let mut em = EventModule::new(EventModuleType::Core);
        // Set combo0=Or(2), combo1=And(0), combo2=AndNot(1).
        let val = 2 | (0 << 8) | (1 << 16);
        assert!(em.write_register(0x4404, val));
        assert_eq!(em.read_register(0x4404), Some(val));
    }

    #[test]
    fn test_register_group_enable_rw() {
        let mut em = EventModule::new(EventModuleType::Core);
        // Group 0 enable register.
        assert!(em.write_register(0x4500, 0x0003));
        assert_eq!(em.read_register(0x4500), Some(0x0003));
        // Group 1.
        assert!(em.write_register(0x4504, 0x000F));
        assert_eq!(em.read_register(0x4504), Some(0x000F));
    }

    #[test]
    fn test_register_port_selection_rw() {
        let mut em = EventModule::new(EventModuleType::Core);
        assert!(em.write_register(0x4FF0, 0x2A_00_23_05));
        assert_eq!(em.read_register(0x4FF0), Some(0x2A_00_23_05));
    }

    #[test]
    fn test_register_unknown_offset() {
        let em = EventModule::new(EventModuleType::Core);
        assert_eq!(em.read_register(0x9999), None);
    }

    #[test]
    fn test_write_unknown_offset() {
        let mut em = EventModule::new(EventModuleType::Core);
        assert!(!em.write_register(0x9999, 42));
    }

    // -- User event generation via register write --

    #[test]
    fn test_user_event_generate_register() {
        let mut em = EventModule::new(EventModuleType::Core);
        // Generate user event 0 (event ID 124 for core module).
        em.write_register(0x4008, 124);
        assert!(em.is_event_active(124));
    }

    // -- Reset tests --

    #[test]
    fn test_reset() {
        let mut em = EventModule::new(EventModuleType::Core);
        em.generate_event(5);
        em.generate_event(42);
        em.configure_broadcast(0, 42);
        em.configure_combo(0, 5, 10, ComboLogic::Or);
        em.configure_group(0, 0x0003);

        em.reset();

        // Status should be reset (only TRUE event set).
        assert_eq!(em.read_event_status(0), 0x2);
        assert!(!em.is_event_active(5));
        assert!(!em.is_event_active(42));

        // Pending should be empty.
        assert!(em.pending_events().is_empty());

        // Group event should be reset to default mask.
        let core_group0_default = 0x0000_0FFF;
        assert_eq!(em.group_events.groups[0].enable_mask, core_group0_default);

        // Combo and broadcast should be reset.
        assert_eq!(em.combo_events.combos[0].input_a, 0);
        assert_eq!(em.broadcast.channels[0].event, 0);
    }

    // -- Module type property tests --

    #[test]
    fn test_module_type_properties() {
        assert_eq!(EventModuleType::Core.num_events(), 128);
        assert_eq!(EventModuleType::Memory.num_events(), 128);
        assert_eq!(EventModuleType::Pl.num_events(), 128);
        assert_eq!(EventModuleType::MemTile.num_events(), 161);

        assert_eq!(EventModuleType::Core.event_id_width(), 7);
        assert_eq!(EventModuleType::MemTile.event_id_width(), 8);

        assert!(EventModuleType::Core.has_port_events());
        assert!(!EventModuleType::Memory.has_port_events());
        assert!(EventModuleType::Pl.has_port_events());
        assert!(EventModuleType::MemTile.has_port_events());
    }

    #[test]
    fn test_combo_event_base_ids() {
        assert_eq!(EventModuleType::Core.combo_event_base(), 9);
        assert_eq!(EventModuleType::Memory.combo_event_base(), 7);
        assert_eq!(EventModuleType::Pl.combo_event_base(), 7);
        assert_eq!(EventModuleType::MemTile.combo_event_base(), 9);
    }

    #[test]
    fn test_broadcast_event_base_ids() {
        assert_eq!(EventModuleType::Core.broadcast_event_base(), 107);
        assert_eq!(EventModuleType::Memory.broadcast_event_base(), 107);
        assert_eq!(EventModuleType::Pl.broadcast_event_base(), 110);
        assert_eq!(EventModuleType::MemTile.broadcast_event_base(), 142);
    }

    #[test]
    fn test_user_event_base_ids() {
        assert_eq!(EventModuleType::Core.user_event_base(), 124);
        assert_eq!(EventModuleType::Memory.user_event_base(), 124);
        assert_eq!(EventModuleType::Pl.user_event_base(), 126);
        assert_eq!(EventModuleType::MemTile.user_event_base(), 159);
    }

    #[test]
    fn test_port_idle_event_base() {
        assert_eq!(EventModuleType::Core.port_idle_event_base(), Some(74));
        assert_eq!(EventModuleType::Memory.port_idle_event_base(), None);
        assert_eq!(EventModuleType::Pl.port_idle_event_base(), Some(77));
        assert_eq!(EventModuleType::MemTile.port_idle_event_base(), Some(79));
    }

    // -- Edge case: simultaneous generate and clear --

    #[test]
    fn test_generate_then_clear_then_regenerate() {
        let mut em = EventModule::new(EventModuleType::Core);
        em.generate_event(42);
        assert!(em.is_event_active(42));
        em.clear_event(42);
        assert!(!em.is_event_active(42));
        em.generate_event(42);
        assert!(em.is_event_active(42));
    }

    // -- Different module types have correct event counts --

    #[test]
    fn test_different_modules_correct_counts() {
        // Core: 9 groups, 128 events, 4 status regs
        let em = EventModule::new(EventModuleType::Core);
        assert_eq!(em.group_events.count(), 9);
        assert_eq!(em.event_status.len(), 4);

        // Memory: 8 groups, 128 events, 4 status regs
        let em = EventModule::new(EventModuleType::Memory);
        assert_eq!(em.group_events.count(), 8);
        assert_eq!(em.event_status.len(), 4);

        // PL: 6 groups, 128 events, 4 status regs
        let em = EventModule::new(EventModuleType::Pl);
        assert_eq!(em.group_events.count(), 6);
        assert_eq!(em.event_status.len(), 4);

        // MemTile: 9 groups, 161 events, 6 status regs
        let em = EventModule::new(EventModuleType::MemTile);
        assert_eq!(em.group_events.count(), 9);
        assert_eq!(em.event_status.len(), 6);
    }
}
