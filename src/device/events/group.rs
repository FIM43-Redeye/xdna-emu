// Group event logic for AIE2 event subsystem.
//
// Each module has a set of group events. A group event is the OR of multiple
// related sub-events, gated by an enable mask. When any enabled sub-event
// fires, the corresponding group event fires.
//
// Per aie-rt: each group has a configurable enable mask register. The mask
// selects which sub-events contribute to the group event output. The reset
// value for each mask enables all sub-events (per xaiemlgbl_reginit.c).
//
// Register layout (per group, 4-byte spacing):
//   BaseGroupEventRegOff + (group_index * 4)
//
// Core module: 9 group events
// Memory module: 8 group events
// PL/NoC module: 6 group events
// MemTile module: 9 group events

/// Configuration and state for a single group event.
///
/// Mirrors XAie_EventGroup from aie-rt. The enable_mask controls which
/// sub-events are OR'd together to produce the group event output.
#[derive(Debug, Clone)]
pub struct GroupEvent {
    /// Which sub-events are enabled (bitmask).
    /// When any enabled sub-event fires, the group event fires.
    pub enable_mask: u32,

    /// Reset/default value for the enable mask.
    /// Per aie-rt, each group's reset value enables all sub-events.
    pub reset_mask: u32,
}

impl GroupEvent {
    /// Create a new group event with the given reset mask.
    /// Starts with all sub-events enabled (reset value).
    pub fn new(reset_mask: u32) -> Self {
        Self { enable_mask: reset_mask, reset_mask }
    }

    /// Configure which sub-events contribute to this group event.
    pub fn configure(&mut self, mask: u32) {
        self.enable_mask = mask;
    }

    /// Reset the enable mask to the default value.
    pub fn reset(&mut self) {
        self.enable_mask = self.reset_mask;
    }

    /// Check if any of the given active sub-events (bitmask) triggers
    /// this group event.
    ///
    /// The sub-event bits are relative to this group's sub-event space.
    /// Returns true if at least one enabled sub-event is active.
    pub fn is_triggered(&self, active_sub_events: u32) -> bool {
        (self.enable_mask & active_sub_events) != 0
    }

    /// Read the enable mask register value.
    pub fn read_register(&self) -> u32 {
        self.enable_mask
    }

    /// Write the enable mask register value.
    pub fn write_register(&mut self, value: u32) {
        self.enable_mask = value;
    }
}

/// Group event definitions per module type.
///
/// Derived from aie-rt xaiemlgbl_reginit.c AieMlCoreGroupEvent,
/// AieMlMemGroupEvent, AieMlPlGroupEvent, AieMlMemTileGroupEvent.
#[derive(Debug, Clone)]
pub struct GroupEventConfig {
    /// The group events for this module.
    pub groups: Vec<GroupEvent>,

    /// The hardware event ID for each group event.
    /// Maps group index -> hardware event ID in this module's event space.
    pub group_event_ids: Vec<u8>,
}

impl GroupEventConfig {
    /// Create core module group events (9 groups).
    ///
    /// Per aie-rt AieMlCoreGroupEvent:
    ///   Group 0 (EVENT_GROUP_0_CORE, hw_id 2): mask 0x0000_0FFF
    ///   Group 1 (EVENT_GROUP_PC_EVENT_CORE, hw_id 15): mask 0x0000_003F
    ///   Group 2 (EVENT_GROUP_CORE_STALL_CORE, hw_id 22): mask 0x0000_01FF
    ///   Group 3 (EVENT_GROUP_CORE_PROGRAM_FLOW_CORE, hw_id 32): mask 0x0000_1FFF
    ///   Group 4 (EVENT_GROUP_ERRORS_0_CORE, hw_id 46): mask 0x01FF_FFFF
    ///   Group 5 (EVENT_GROUP_ERRORS_1_CORE, hw_id 47): mask 0x01FF_FFFF
    ///   Group 6 (EVENT_GROUP_STREAM_SWITCH_CORE, hw_id 73): mask 0xFFFF_FFFF
    ///   Group 7 (EVENT_GROUP_BROADCAST_CORE, hw_id 106): mask 0x0000_FFFF
    ///   Group 8 (EVENT_GROUP_USER_EVENT_CORE, hw_id 123): mask 0x0000_000F
    pub fn core() -> Self {
        Self {
            groups: vec![
                GroupEvent::new(0x0000_0FFF),
                GroupEvent::new(0x0000_003F),
                GroupEvent::new(0x0000_01FF),
                GroupEvent::new(0x0000_1FFF),
                GroupEvent::new(0x01FF_FFFF),
                GroupEvent::new(0x01FF_FFFF),
                GroupEvent::new(0xFFFF_FFFF),
                GroupEvent::new(0x0000_FFFF),
                GroupEvent::new(0x0000_000F),
            ],
            group_event_ids: vec![2, 15, 22, 32, 46, 47, 73, 106, 123],
        }
    }

    /// Create memory module group events (8 groups).
    ///
    /// Per aie-rt AieMlMemGroupEvent:
    ///   Group 0 (EVENT_GROUP_0_MEM, hw_id 2): mask 0x0000_03FF
    ///   Group 1 (EVENT_GROUP_WATCHPOINT_MEM, hw_id 15): mask 0x0000_0003
    ///   Group 2 (EVENT_GROUP_DMA_ACTIVITY_MEM, hw_id 18): mask 0x00FF_FFFF
    ///   Group 3 (EVENT_GROUP_LOCK_MEM, hw_id 43): mask 0xFFFF_FFFF
    ///   Group 4 (EVENT_GROUP_MEMORY_CONFLICT_MEM, hw_id 76): mask 0x0000_00FF
    ///   Group 5 (EVENT_GROUP_ERRORS_MEM, hw_id 86): mask 0x0000_FFFF
    ///   Group 6 (EVENT_GROUP_BROADCAST_MEM, hw_id 106): mask 0x0000_FFFF
    ///   Group 7 (EVENT_GROUP_USER_EVENT_MEM, hw_id 123): mask 0x0000_000F
    pub fn memory() -> Self {
        Self {
            groups: vec![
                GroupEvent::new(0x0000_03FF),
                GroupEvent::new(0x0000_0003),
                GroupEvent::new(0x00FF_FFFF),
                GroupEvent::new(0xFFFF_FFFF),
                GroupEvent::new(0x0000_00FF),
                GroupEvent::new(0x0000_FFFF),
                GroupEvent::new(0x0000_FFFF),
                GroupEvent::new(0x0000_000F),
            ],
            group_event_ids: vec![2, 15, 18, 43, 76, 86, 106, 123],
        }
    }

    /// Create PL/NoC module group events (6 groups).
    ///
    /// Per aie-rt AieMlPlGroupEvent:
    ///   Group 0 (EVENT_GROUP_0_PL, hw_id 2): mask 0x0000_03FF
    ///   Group 1 (EVENT_GROUP_DMA_ACTIVITY_PL, hw_id 13): mask 0x00FF_FFFF
    ///   Group 2 (EVENT_GROUP_LOCK_PL, hw_id 38): mask 0x00FF_FFFF
    ///   Group 3 (EVENT_GROUP_ERRORS_PL, hw_id 63): mask 0x0000_0FFF
    ///   Group 4 (EVENT_GROUP_STREAM_SWITCH_PL, hw_id 76): mask 0xFFFF_FFFF
    ///   Group 5 (EVENT_GROUP_BROADCAST_A_PL, hw_id 109): mask 0x0000_FFFF
    pub fn pl() -> Self {
        Self {
            groups: vec![
                GroupEvent::new(0x0000_03FF),
                GroupEvent::new(0x00FF_FFFF),
                GroupEvent::new(0x00FF_FFFF),
                GroupEvent::new(0x0000_0FFF),
                GroupEvent::new(0xFFFF_FFFF),
                GroupEvent::new(0x0000_FFFF),
            ],
            group_event_ids: vec![2, 13, 38, 63, 76, 109],
        }
    }

    /// Create mem tile module group events (9 groups).
    ///
    /// Per aie-rt AieMlMemTileGroupEvent:
    ///   Group 0 (EVENT_GROUP_0_MEM_TILE, hw_id 2): mask 0x0000_0FFF
    ///   Group 1 (EVENT_GROUP_WATCHPOINT_MEM_TILE, hw_id 15): mask 0x0000_000F
    ///   Group 2 (EVENT_GROUP_DMA_ACTIVITY_MEM_TILE, hw_id 20): mask 0x00FF_FFFF
    ///   Group 3 (EVENT_GROUP_LOCK_MEM_TILE, hw_id 45): mask 0xFFFF_FFFF
    ///   Group 4 (EVENT_GROUP_STREAM_SWITCH_MEM_TILE, hw_id 78): mask 0xFFFF_FFFF
    ///   Group 5 (EVENT_GROUP_MEMORY_CONFLICT_MEM_TILE, hw_id 111): mask 0x0000_FFFF
    ///   Group 6 (EVENT_GROUP_ERRORS_MEM_TILE, hw_id 128): mask 0x0000_0FFF
    ///   Group 7 (EVENT_GROUP_BROADCAST_MEM_TILE, hw_id 141): mask 0x0000_FFFF
    ///   Group 8 (EVENT_GROUP_USER_EVENT_MEM_TILE, hw_id 158): mask 0x0000_0003
    pub fn mem_tile() -> Self {
        Self {
            groups: vec![
                GroupEvent::new(0x0000_0FFF),
                GroupEvent::new(0x0000_000F),
                GroupEvent::new(0x00FF_FFFF),
                GroupEvent::new(0xFFFF_FFFF),
                GroupEvent::new(0xFFFF_FFFF),
                GroupEvent::new(0x0000_FFFF),
                GroupEvent::new(0x0000_0FFF),
                GroupEvent::new(0x0000_FFFF),
                GroupEvent::new(0x0000_0003),
            ],
            group_event_ids: vec![2, 15, 20, 45, 78, 111, 128, 141, 158],
        }
    }

    /// Number of group events in this module.
    pub fn count(&self) -> usize {
        self.groups.len()
    }

    /// Find a group event by its hardware event ID.
    /// Returns the group index if found.
    pub fn find_by_event_id(&self, event_id: u8) -> Option<usize> {
        self.group_event_ids.iter().position(|&id| id == event_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_event_new() {
        let ge = GroupEvent::new(0x0FFF);
        assert_eq!(ge.enable_mask, 0x0FFF);
        assert_eq!(ge.reset_mask, 0x0FFF);
    }

    #[test]
    fn test_group_event_configure() {
        let mut ge = GroupEvent::new(0x0FFF);
        ge.configure(0x0003);
        assert_eq!(ge.enable_mask, 0x0003);
        // Reset mask unchanged.
        assert_eq!(ge.reset_mask, 0x0FFF);
    }

    #[test]
    fn test_group_event_reset() {
        let mut ge = GroupEvent::new(0x0FFF);
        ge.configure(0x0003);
        ge.reset();
        assert_eq!(ge.enable_mask, 0x0FFF);
    }

    #[test]
    fn test_group_event_triggered_all_enabled() {
        let ge = GroupEvent::new(0x000F);
        assert!(ge.is_triggered(0x0001));
        assert!(ge.is_triggered(0x0008));
        // Bit 4 is outside the mask.
        assert!(!ge.is_triggered(0x0010));
    }

    #[test]
    fn test_group_event_triggered_some_disabled() {
        let mut ge = GroupEvent::new(0x000F);
        ge.configure(0x0003);
        assert!(ge.is_triggered(0x0001));
        // Bit 2 now disabled.
        assert!(!ge.is_triggered(0x0004));
        // Bit 1 enabled even though bit 2 disabled.
        assert!(ge.is_triggered(0x0006));
    }

    #[test]
    fn test_group_event_triggered_none_active() {
        let ge = GroupEvent::new(0x000F);
        assert!(!ge.is_triggered(0x0000));
    }

    #[test]
    fn test_group_event_register_rw() {
        let mut ge = GroupEvent::new(0x0FFF);
        assert_eq!(ge.read_register(), 0x0FFF);
        ge.write_register(0x0042);
        assert_eq!(ge.read_register(), 0x0042);
    }

    #[test]
    fn test_core_group_config() {
        let cfg = GroupEventConfig::core();
        assert_eq!(cfg.count(), 9);
        assert_eq!(cfg.group_event_ids[0], 2);
        assert_eq!(cfg.group_event_ids[8], 123);
        assert_eq!(cfg.groups[0].reset_mask, 0x0000_0FFF);
    }

    #[test]
    fn test_memory_group_config() {
        let cfg = GroupEventConfig::memory();
        assert_eq!(cfg.count(), 8);
        assert_eq!(cfg.group_event_ids[0], 2);
        assert_eq!(cfg.group_event_ids[7], 123);
    }

    #[test]
    fn test_pl_group_config() {
        let cfg = GroupEventConfig::pl();
        assert_eq!(cfg.count(), 6);
        assert_eq!(cfg.group_event_ids[0], 2);
        assert_eq!(cfg.group_event_ids[5], 109);
    }

    #[test]
    fn test_mem_tile_group_config() {
        let cfg = GroupEventConfig::mem_tile();
        assert_eq!(cfg.count(), 9);
        assert_eq!(cfg.group_event_ids[0], 2);
        assert_eq!(cfg.group_event_ids[8], 158);
    }

    #[test]
    fn test_find_by_event_id() {
        let cfg = GroupEventConfig::core();
        assert_eq!(cfg.find_by_event_id(2), Some(0));
        assert_eq!(cfg.find_by_event_id(22), Some(2));
        assert_eq!(cfg.find_by_event_id(123), Some(8));
        assert_eq!(cfg.find_by_event_id(99), None);
    }
}
