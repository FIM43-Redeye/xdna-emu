// Port event logic for AIE2 event subsystem.
//
// Each module with stream switch access has 8 stream switch event port
// selection slots. Each slot maps a logical event port to a physical
// stream switch port. The hardware then generates PORT_IDLE, PORT_RUNNING,
// PORT_STALLED, and PORT_TLAST events for that port.
//
// Per aie-rt xaie_events.c _XAie_EventSelectStrmPortConfig():
// - Each slot selects a physical port index and master/slave direction.
// - 4 slots are packed per 32-bit register (8 bits per slot).
// - Within each 8-bit slot: [4:0] port_id, [5] master(1)/slave(0).
//
// Register layout (from xaiemlgbl_params.h):
//   STREAM_SWITCH_EVENT_PORT_SELECTION_0: 4 slots packed
//   STREAM_SWITCH_EVENT_PORT_SELECTION_1: 4 slots packed
//
// Port events in the event space (core module example):
//   PORT_IDLE_0    = 74   (base + slot*4 + 0)
//   PORT_RUNNING_0 = 75   (base + slot*4 + 1)
//   PORT_STALLED_0 = 76   (base + slot*4 + 2)
//   PORT_TLAST_0   = 77   (base + slot*4 + 3)
//   PORT_IDLE_1    = 78
//   ... etc for 8 slots
//
// Memory module of AIE tiles does NOT have stream switch event ports
// (NumStrmPortSelectIds = XAIE_FEATURE_UNAVAILABLE).

/// Number of stream switch event port selection slots per module.
/// Per aie-rt: NumStrmPortSelectIds = 8U for core, PL, and memtile modules.
pub const NUM_PORT_EVENT_SLOTS: usize = 8;

/// Number of port event types per slot (Idle, Running, Stalled, Tlast).
pub const PORT_EVENTS_PER_SLOT: usize = 4;

/// Port activity event type.
///
/// Each selected port generates 4 events, one for each activity state.
/// These correspond to the 4 consecutive event IDs per slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PortEventType {
    /// Port is idle (no data transfer).
    Idle = 0,
    /// Port is actively transferring data.
    Running = 1,
    /// Port is stalled (backpressure or starvation).
    Stalled = 2,
    /// TLAST seen on this port (end of packet/transfer).
    Tlast = 3,
}

impl PortEventType {
    /// All port event types for iteration.
    pub const ALL: [PortEventType; 4] =
        [PortEventType::Idle, PortEventType::Running, PortEventType::Stalled, PortEventType::Tlast];
}

/// Configuration for a single port event selection slot.
///
/// Maps a logical event port to a physical stream switch port.
#[derive(Debug, Clone, Copy)]
pub struct PortEventSlot {
    /// Physical stream switch port index (5 bits, 0-31).
    pub port_id: u8,

    /// true = master port, false = slave port.
    /// Per aie-rt: XAIE_STRMSW_MASTER = 1, XAIE_STRMSW_SLAVE = 0.
    pub is_master: bool,
}

impl PortEventSlot {
    /// Create a new slot in reset state (slave port 0).
    pub fn new() -> Self {
        Self { port_id: 0, is_master: false }
    }

    /// Configure this slot.
    pub fn configure(&mut self, port_id: u8, is_master: bool) {
        self.port_id = port_id & 0x1F; // 5-bit port ID
        self.is_master = is_master;
    }

    /// Pack into the 8-bit register field: [4:0] port_id, [5] master/slave.
    pub fn to_register_field(&self) -> u8 {
        let ms_bit = if self.is_master { 1 << 5 } else { 0 };
        (self.port_id & 0x1F) | ms_bit
    }

    /// Unpack from the 8-bit register field.
    pub fn from_register_field(val: u8) -> Self {
        Self { port_id: val & 0x1F, is_master: (val & (1 << 5)) != 0 }
    }

    /// Reset to default state.
    pub fn reset(&mut self) {
        self.port_id = 0;
        self.is_master = false;
    }
}

impl Default for PortEventSlot {
    fn default() -> Self {
        Self::new()
    }
}

/// Port event configuration for one module.
///
/// Manages 8 port event selection slots and provides the register
/// interface for STREAM_SWITCH_EVENT_PORT_SELECTION registers.
#[derive(Debug, Clone)]
pub struct PortEventConfig {
    /// The 8 port event selection slots.
    pub slots: [PortEventSlot; NUM_PORT_EVENT_SLOTS],

    /// Whether this module supports port events.
    /// Memory module of AIE tiles does not (XAIE_FEATURE_UNAVAILABLE).
    pub available: bool,
}

impl PortEventConfig {
    /// Create a new port event configuration.
    ///
    /// `available` should be false for memory modules of AIE tiles,
    /// which do not have stream switch event port selection.
    pub fn new(available: bool) -> Self {
        Self { slots: std::array::from_fn(|_| PortEventSlot::new()), available }
    }

    /// Configure a port event slot.
    pub fn configure_slot(&mut self, slot: usize, port_id: u8, is_master: bool) {
        if slot < NUM_PORT_EVENT_SLOTS && self.available {
            self.slots[slot].configure(port_id, is_master);
        }
    }

    /// Read one of the two port selection registers.
    ///
    /// Register 0 contains slots 0-3, register 1 contains slots 4-7.
    /// Per aie-rt: StrmPortSelectIdsPerReg = 4U, PortIdOff = 8U.
    pub fn read_register(&self, reg_index: usize) -> u32 {
        if !self.available || reg_index > 1 {
            return 0;
        }
        let base = reg_index * 4;
        let mut val: u32 = 0;
        for i in 0..4 {
            let slot_idx = base + i;
            if slot_idx < NUM_PORT_EVENT_SLOTS {
                val |= (self.slots[slot_idx].to_register_field() as u32) << (i * 8);
            }
        }
        val
    }

    /// Write one of the two port selection registers.
    pub fn write_register(&mut self, reg_index: usize, value: u32) {
        if !self.available || reg_index > 1 {
            return;
        }
        let base = reg_index * 4;
        for i in 0..4 {
            let slot_idx = base + i;
            if slot_idx < NUM_PORT_EVENT_SLOTS {
                let field = ((value >> (i * 8)) & 0xFF) as u8;
                self.slots[slot_idx] = PortEventSlot::from_register_field(field);
            }
        }
    }

    /// Get the hardware event ID for a port event, given the base event ID.
    ///
    /// For slot N and event type T:
    ///   event_id = port_idle_event_base + (slot * 4) + T
    ///
    /// Returns None if port events are unavailable.
    pub fn event_id_for_slot(
        &self,
        slot: usize,
        event_type: PortEventType,
        port_idle_base: u8,
    ) -> Option<u8> {
        if !self.available || slot >= NUM_PORT_EVENT_SLOTS {
            return None;
        }
        Some(
            port_idle_base
                .wrapping_add((slot * PORT_EVENTS_PER_SLOT) as u8)
                .wrapping_add(event_type as u8),
        )
    }

    /// Reset all slots to default state.
    pub fn reset(&mut self) {
        for slot in &mut self.slots {
            slot.reset();
        }
    }
}

impl Default for PortEventConfig {
    fn default() -> Self {
        Self::new(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- PortEventSlot tests --

    #[test]
    fn test_slot_new() {
        let slot = PortEventSlot::new();
        assert_eq!(slot.port_id, 0);
        assert!(!slot.is_master);
    }

    #[test]
    fn test_slot_configure() {
        let mut slot = PortEventSlot::new();
        slot.configure(5, true);
        assert_eq!(slot.port_id, 5);
        assert!(slot.is_master);
    }

    #[test]
    fn test_slot_configure_truncates_port_id() {
        let mut slot = PortEventSlot::new();
        // Port ID is 5 bits, so 0xFF should become 0x1F.
        slot.configure(0xFF, false);
        assert_eq!(slot.port_id, 0x1F);
    }

    #[test]
    fn test_slot_register_field_slave() {
        let mut slot = PortEventSlot::new();
        slot.configure(7, false);
        assert_eq!(slot.to_register_field(), 7);
    }

    #[test]
    fn test_slot_register_field_master() {
        let mut slot = PortEventSlot::new();
        slot.configure(7, true);
        // Bit 5 set for master.
        assert_eq!(slot.to_register_field(), 7 | (1 << 5));
    }

    #[test]
    fn test_slot_from_register_field() {
        let slot = PortEventSlot::from_register_field(0x25); // port 5, master
        assert_eq!(slot.port_id, 5);
        assert!(slot.is_master);

        let slot = PortEventSlot::from_register_field(0x03); // port 3, slave
        assert_eq!(slot.port_id, 3);
        assert!(!slot.is_master);
    }

    #[test]
    fn test_slot_register_field_roundtrip() {
        let mut slot = PortEventSlot::new();
        slot.configure(13, true);
        let field = slot.to_register_field();
        let recovered = PortEventSlot::from_register_field(field);
        assert_eq!(recovered.port_id, 13);
        assert!(recovered.is_master);
    }

    #[test]
    fn test_slot_reset() {
        let mut slot = PortEventSlot::new();
        slot.configure(10, true);
        slot.reset();
        assert_eq!(slot.port_id, 0);
        assert!(!slot.is_master);
    }

    // -- PortEventConfig tests --

    #[test]
    fn test_config_new_available() {
        let cfg = PortEventConfig::new(true);
        assert!(cfg.available);
        assert_eq!(cfg.slots.len(), 8);
        for slot in &cfg.slots {
            assert_eq!(slot.port_id, 0);
            assert!(!slot.is_master);
        }
    }

    #[test]
    fn test_config_new_unavailable() {
        let cfg = PortEventConfig::new(false);
        assert!(!cfg.available);
    }

    #[test]
    fn test_config_configure_slot() {
        let mut cfg = PortEventConfig::new(true);
        cfg.configure_slot(3, 7, true);
        assert_eq!(cfg.slots[3].port_id, 7);
        assert!(cfg.slots[3].is_master);
    }

    #[test]
    fn test_config_configure_slot_unavailable() {
        let mut cfg = PortEventConfig::new(false);
        cfg.configure_slot(0, 7, true);
        // Should be ignored when unavailable.
        assert_eq!(cfg.slots[0].port_id, 0);
    }

    #[test]
    fn test_config_configure_slot_out_of_bounds() {
        let mut cfg = PortEventConfig::new(true);
        cfg.configure_slot(8, 7, true); // Should be silently ignored.
    }

    #[test]
    fn test_config_read_register_0() {
        let mut cfg = PortEventConfig::new(true);
        cfg.configure_slot(0, 3, false); // 0x03
        cfg.configure_slot(1, 7, true); // 0x27
        cfg.configure_slot(2, 0, false); // 0x00
        cfg.configure_slot(3, 15, true); // 0x2F

        let reg = cfg.read_register(0);
        assert_eq!(reg & 0xFF, 0x03);
        assert_eq!((reg >> 8) & 0xFF, 0x27);
        assert_eq!((reg >> 16) & 0xFF, 0x00);
        assert_eq!((reg >> 24) & 0xFF, 0x2F);
    }

    #[test]
    fn test_config_read_register_1() {
        let mut cfg = PortEventConfig::new(true);
        cfg.configure_slot(4, 1, false); // 0x01
        cfg.configure_slot(5, 2, true); // 0x22

        let reg = cfg.read_register(1);
        assert_eq!(reg & 0xFF, 0x01);
        assert_eq!((reg >> 8) & 0xFF, 0x22);
    }

    #[test]
    fn test_config_read_register_unavailable() {
        let cfg = PortEventConfig::new(false);
        assert_eq!(cfg.read_register(0), 0);
    }

    #[test]
    fn test_config_read_register_out_of_bounds() {
        let cfg = PortEventConfig::new(true);
        assert_eq!(cfg.read_register(2), 0);
    }

    #[test]
    fn test_config_write_register() {
        let mut cfg = PortEventConfig::new(true);
        // Slot 0: port 5 slave (0x05), Slot 1: port 3 master (0x23),
        // Slot 2: port 0 slave (0x00), Slot 3: port 10 master (0x2A)
        cfg.write_register(0, 0x2A_00_23_05);

        assert_eq!(cfg.slots[0].port_id, 5);
        assert!(!cfg.slots[0].is_master);
        assert_eq!(cfg.slots[1].port_id, 3);
        assert!(cfg.slots[1].is_master);
        assert_eq!(cfg.slots[2].port_id, 0);
        assert!(!cfg.slots[2].is_master);
        assert_eq!(cfg.slots[3].port_id, 10);
        assert!(cfg.slots[3].is_master);
    }

    #[test]
    fn test_config_write_register_unavailable() {
        let mut cfg = PortEventConfig::new(false);
        cfg.write_register(0, 0xFFFF_FFFF);
        // Should be ignored.
        assert_eq!(cfg.slots[0].port_id, 0);
    }

    #[test]
    fn test_config_register_roundtrip() {
        let mut cfg = PortEventConfig::new(true);
        cfg.configure_slot(0, 3, false);
        cfg.configure_slot(1, 7, true);
        cfg.configure_slot(2, 12, false);
        cfg.configure_slot(3, 15, true);

        let reg0 = cfg.read_register(0);

        let mut cfg2 = PortEventConfig::new(true);
        cfg2.write_register(0, reg0);

        assert_eq!(cfg2.slots[0].port_id, 3);
        assert!(!cfg2.slots[0].is_master);
        assert_eq!(cfg2.slots[1].port_id, 7);
        assert!(cfg2.slots[1].is_master);
        assert_eq!(cfg2.slots[2].port_id, 12);
        assert!(!cfg2.slots[2].is_master);
        assert_eq!(cfg2.slots[3].port_id, 15);
        assert!(cfg2.slots[3].is_master);
    }

    #[test]
    fn test_event_id_for_slot() {
        let cfg = PortEventConfig::new(true);
        let base = 74; // PORT_IDLE_0 for core module

        assert_eq!(cfg.event_id_for_slot(0, PortEventType::Idle, base), Some(74));
        assert_eq!(cfg.event_id_for_slot(0, PortEventType::Running, base), Some(75));
        assert_eq!(cfg.event_id_for_slot(0, PortEventType::Stalled, base), Some(76));
        assert_eq!(cfg.event_id_for_slot(0, PortEventType::Tlast, base), Some(77));

        assert_eq!(cfg.event_id_for_slot(1, PortEventType::Idle, base), Some(78));
        assert_eq!(cfg.event_id_for_slot(7, PortEventType::Tlast, base), Some(74 + 7 * 4 + 3));
    }

    #[test]
    fn test_event_id_for_slot_unavailable() {
        let cfg = PortEventConfig::new(false);
        assert_eq!(cfg.event_id_for_slot(0, PortEventType::Idle, 74), None);
    }

    #[test]
    fn test_event_id_for_slot_out_of_bounds() {
        let cfg = PortEventConfig::new(true);
        assert_eq!(cfg.event_id_for_slot(8, PortEventType::Idle, 74), None);
    }

    #[test]
    fn test_config_reset() {
        let mut cfg = PortEventConfig::new(true);
        cfg.configure_slot(0, 5, true);
        cfg.configure_slot(7, 12, false);
        cfg.reset();

        for slot in &cfg.slots {
            assert_eq!(slot.port_id, 0);
            assert!(!slot.is_master);
        }
    }
}
