// Broadcast event logic for AIE2 event subsystem.
//
// Each module has 16 broadcast channels. A broadcast channel is configured
// with a local event ID: when that event fires, the broadcast channel
// propagates it to neighboring tiles in all four directions (south, west,
// north, east), subject to directional blocking masks.
//
// Per aie-rt xaie_events.c _XAie_EventBroadcastConfig():
// - Each channel has one register: BaseBroadcastRegOff + (channel * 4).
// - The register holds the 7-bit (or 8-bit for memtile) event ID to broadcast.
// - Writing 0 (EVENT_NONE) effectively disables the channel.
//
// Broadcast blocking uses SET/CLR/VALUE register triplets per direction.
// Per aie-rt XAie_EventBroadcastBlockDir():
// - Block SET register: writing a bit sets that channel's block flag.
// - Block CLR register: writing a bit clears that channel's block flag.
// - Block VALUE register: reads the current block state for that direction.
//
// Register layout (core module, from xaiemlgbl_params.h):
//   EVENT_BROADCAST0:                    0x34010  (+ channel * 4)
//   EVENT_BROADCAST_BLOCK_SOUTH_SET:     0x34050
//   EVENT_BROADCAST_BLOCK_SOUTH_CLR:     0x34054
//   EVENT_BROADCAST_BLOCK_SOUTH_VALUE:   0x34058
//   EVENT_BROADCAST_BLOCK_WEST_SET:      0x34060  (stride 0x10 per direction)
//   EVENT_BROADCAST_BLOCK_WEST_CLR:      0x34064
//   EVENT_BROADCAST_BLOCK_WEST_VALUE:    0x34068
//   ... (north at +0x20, east at +0x30 from south base)
//
// Direction indices (per aie-rt, used as bit positions in Dir parameter):
//   0 = South
//   1 = West
//   2 = North
//   3 = East

/// Number of broadcast channels per module (all module types).
/// Per aie-rt: NumBroadcastIds = 16U for all AIEML modules.
pub const NUM_BROADCAST_CHANNELS: usize = 16;

/// Broadcast direction flags, matching aie-rt XAIE_EVENT_BROADCAST_* values.
/// These are bit positions used to index the block registers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum BroadcastDir {
    South = 0,
    West = 1,
    North = 2,
    East = 3,
}

impl BroadcastDir {
    /// Number of directions.
    pub const COUNT: usize = 4;

    /// All directions for iteration.
    pub const ALL: [BroadcastDir; 4] = [
        BroadcastDir::South,
        BroadcastDir::West,
        BroadcastDir::North,
        BroadcastDir::East,
    ];

    /// Convert from direction index (0-3).
    pub fn from_index(idx: usize) -> Option<Self> {
        match idx {
            0 => Some(BroadcastDir::South),
            1 => Some(BroadcastDir::West),
            2 => Some(BroadcastDir::North),
            3 => Some(BroadcastDir::East),
            _ => None,
        }
    }
}

/// A single broadcast channel.
#[derive(Debug, Clone)]
pub struct BroadcastChannel {
    /// The local event ID that triggers this broadcast channel.
    /// 0 = disabled (EVENT_NONE).
    pub event: u8,
}

impl BroadcastChannel {
    /// Create a new broadcast channel in disabled state.
    pub fn new() -> Self {
        Self { event: 0 }
    }
}

impl Default for BroadcastChannel {
    fn default() -> Self {
        Self::new()
    }
}

/// Broadcast event configuration for one module.
///
/// Contains 16 broadcast channels and directional blocking masks.
/// The blocking masks are per-direction bitmaps where bit N controls
/// whether broadcast channel N is blocked in that direction.
#[derive(Debug, Clone)]
pub struct BroadcastConfig {
    /// The 16 broadcast channels.
    pub channels: [BroadcastChannel; NUM_BROADCAST_CHANNELS],

    /// Per-direction blocking masks. Index by BroadcastDir as usize.
    /// Bit N = 1 means channel N is blocked in that direction.
    pub block_mask: [u16; BroadcastDir::COUNT],
}

impl BroadcastConfig {
    /// Create a new broadcast configuration in reset state.
    /// All channels disabled (event=0), no blocking.
    pub fn new() -> Self {
        Self {
            channels: std::array::from_fn(|_| BroadcastChannel::new()),
            block_mask: [0; BroadcastDir::COUNT],
        }
    }

    /// Configure a broadcast channel to broadcast the given event.
    /// Writing 0 effectively disables the channel.
    pub fn configure_channel(&mut self, channel: usize, event: u8) {
        if channel < NUM_BROADCAST_CHANNELS {
            self.channels[channel].event = event;
        }
    }

    /// Read the broadcast channel register value (7-bit event ID).
    pub fn read_channel(&self, channel: usize) -> u32 {
        if channel < NUM_BROADCAST_CHANNELS {
            self.channels[channel].event as u32
        } else {
            0
        }
    }

    /// Block a broadcast channel in a specific direction.
    ///
    /// Per aie-rt: writing to the SET register sets the block bit.
    pub fn block_channel(&mut self, channel: usize, dir: BroadcastDir) {
        if channel < NUM_BROADCAST_CHANNELS {
            self.block_mask[dir as usize] |= 1 << channel;
        }
    }

    /// Unblock a broadcast channel in a specific direction.
    ///
    /// Per aie-rt: writing to the CLR register clears the block bit.
    pub fn unblock_channel(&mut self, channel: usize, dir: BroadcastDir) {
        if channel < NUM_BROADCAST_CHANNELS {
            self.block_mask[dir as usize] &= !(1 << channel);
        }
    }

    /// Check if a broadcast channel is blocked in a specific direction.
    pub fn is_blocked(&self, channel: usize, dir: BroadcastDir) -> bool {
        if channel < NUM_BROADCAST_CHANNELS {
            (self.block_mask[dir as usize] & (1 << channel)) != 0
        } else {
            true
        }
    }

    /// Write the block SET register for a direction.
    /// Bits that are 1 in `value` set the corresponding block flags.
    pub fn write_block_set(&mut self, dir: BroadcastDir, value: u16) {
        self.block_mask[dir as usize] |= value;
    }

    /// Write the block CLR register for a direction.
    /// Bits that are 1 in `value` clear the corresponding block flags.
    pub fn write_block_clr(&mut self, dir: BroadcastDir, value: u16) {
        self.block_mask[dir as usize] &= !value;
    }

    /// Read the block VALUE register for a direction.
    /// Returns the current block mask.
    pub fn read_block_value(&self, dir: BroadcastDir) -> u16 {
        self.block_mask[dir as usize]
    }

    /// Get the list of directions a broadcast channel can propagate to.
    /// Returns directions that are NOT blocked.
    pub fn allowed_directions(&self, channel: usize) -> Vec<BroadcastDir> {
        if channel >= NUM_BROADCAST_CHANNELS {
            return Vec::new();
        }
        BroadcastDir::ALL
            .iter()
            .filter(|&&dir| !self.is_blocked(channel, dir))
            .copied()
            .collect()
    }

    /// Get pending broadcast events for a given set of currently active events.
    ///
    /// Returns a list of (channel_index, event_id) for channels whose
    /// configured event is in the active set. The caller is responsible
    /// for checking directional blocking before forwarding.
    pub fn pending_broadcasts(&self, is_active: &dyn Fn(u8) -> bool) -> Vec<(usize, u8)> {
        let mut pending = Vec::new();
        for (i, ch) in self.channels.iter().enumerate() {
            if ch.event != 0 && is_active(ch.event) {
                pending.push((i, ch.event));
            }
        }
        pending
    }

    /// Reset all broadcast channels and blocking masks.
    pub fn reset(&mut self) {
        for ch in &mut self.channels {
            ch.event = 0;
        }
        self.block_mask = [0; BroadcastDir::COUNT];
    }
}

impl Default for BroadcastConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_broadcast_channel_new() {
        let ch = BroadcastChannel::new();
        assert_eq!(ch.event, 0);
    }

    #[test]
    fn test_broadcast_config_new() {
        let cfg = BroadcastConfig::new();
        for ch in &cfg.channels {
            assert_eq!(ch.event, 0);
        }
        for mask in &cfg.block_mask {
            assert_eq!(*mask, 0);
        }
    }

    #[test]
    fn test_configure_channel() {
        let mut cfg = BroadcastConfig::new();
        cfg.configure_channel(0, 42);
        assert_eq!(cfg.channels[0].event, 42);
        cfg.configure_channel(15, 100);
        assert_eq!(cfg.channels[15].event, 100);
    }

    #[test]
    fn test_configure_channel_out_of_bounds() {
        let mut cfg = BroadcastConfig::new();
        // Should be silently ignored.
        cfg.configure_channel(16, 42);
        cfg.configure_channel(255, 42);
    }

    #[test]
    fn test_read_channel() {
        let mut cfg = BroadcastConfig::new();
        cfg.configure_channel(3, 55);
        assert_eq!(cfg.read_channel(3), 55);
        assert_eq!(cfg.read_channel(0), 0);
        // Out of bounds returns 0.
        assert_eq!(cfg.read_channel(16), 0);
    }

    #[test]
    fn test_block_channel() {
        let mut cfg = BroadcastConfig::new();
        assert!(!cfg.is_blocked(0, BroadcastDir::South));

        cfg.block_channel(0, BroadcastDir::South);
        assert!(cfg.is_blocked(0, BroadcastDir::South));
        // Other directions unaffected.
        assert!(!cfg.is_blocked(0, BroadcastDir::North));
        assert!(!cfg.is_blocked(0, BroadcastDir::West));
        assert!(!cfg.is_blocked(0, BroadcastDir::East));
    }

    #[test]
    fn test_unblock_channel() {
        let mut cfg = BroadcastConfig::new();
        cfg.block_channel(5, BroadcastDir::West);
        assert!(cfg.is_blocked(5, BroadcastDir::West));

        cfg.unblock_channel(5, BroadcastDir::West);
        assert!(!cfg.is_blocked(5, BroadcastDir::West));
    }

    #[test]
    fn test_block_multiple_channels() {
        let mut cfg = BroadcastConfig::new();
        cfg.block_channel(0, BroadcastDir::South);
        cfg.block_channel(3, BroadcastDir::South);
        cfg.block_channel(15, BroadcastDir::South);

        assert!(cfg.is_blocked(0, BroadcastDir::South));
        assert!(!cfg.is_blocked(1, BroadcastDir::South));
        assert!(cfg.is_blocked(3, BroadcastDir::South));
        assert!(cfg.is_blocked(15, BroadcastDir::South));
    }

    #[test]
    fn test_block_set_register() {
        let mut cfg = BroadcastConfig::new();
        // Set bits 0 and 2 for south.
        cfg.write_block_set(BroadcastDir::South, 0x0005);
        assert!(cfg.is_blocked(0, BroadcastDir::South));
        assert!(!cfg.is_blocked(1, BroadcastDir::South));
        assert!(cfg.is_blocked(2, BroadcastDir::South));

        // Additional set should OR with existing.
        cfg.write_block_set(BroadcastDir::South, 0x0002);
        assert!(cfg.is_blocked(0, BroadcastDir::South));
        assert!(cfg.is_blocked(1, BroadcastDir::South));
        assert!(cfg.is_blocked(2, BroadcastDir::South));
    }

    #[test]
    fn test_block_clr_register() {
        let mut cfg = BroadcastConfig::new();
        cfg.write_block_set(BroadcastDir::North, 0x000F);
        assert_eq!(cfg.read_block_value(BroadcastDir::North), 0x000F);

        // Clear bits 0 and 2.
        cfg.write_block_clr(BroadcastDir::North, 0x0005);
        assert_eq!(cfg.read_block_value(BroadcastDir::North), 0x000A);
    }

    #[test]
    fn test_read_block_value() {
        let mut cfg = BroadcastConfig::new();
        cfg.block_channel(0, BroadcastDir::East);
        cfg.block_channel(7, BroadcastDir::East);
        assert_eq!(cfg.read_block_value(BroadcastDir::East), (1 << 0) | (1 << 7));
    }

    #[test]
    fn test_allowed_directions_none_blocked() {
        let cfg = BroadcastConfig::new();
        let dirs = cfg.allowed_directions(0);
        assert_eq!(dirs.len(), 4);
    }

    #[test]
    fn test_allowed_directions_some_blocked() {
        let mut cfg = BroadcastConfig::new();
        cfg.block_channel(0, BroadcastDir::South);
        cfg.block_channel(0, BroadcastDir::East);

        let dirs = cfg.allowed_directions(0);
        assert_eq!(dirs.len(), 2);
        assert!(dirs.contains(&BroadcastDir::West));
        assert!(dirs.contains(&BroadcastDir::North));
        assert!(!dirs.contains(&BroadcastDir::South));
        assert!(!dirs.contains(&BroadcastDir::East));
    }

    #[test]
    fn test_allowed_directions_all_blocked() {
        let mut cfg = BroadcastConfig::new();
        for dir in BroadcastDir::ALL {
            cfg.block_channel(5, dir);
        }
        let dirs = cfg.allowed_directions(5);
        assert!(dirs.is_empty());
    }

    #[test]
    fn test_allowed_directions_out_of_bounds() {
        let cfg = BroadcastConfig::new();
        let dirs = cfg.allowed_directions(16);
        assert!(dirs.is_empty());
    }

    #[test]
    fn test_pending_broadcasts() {
        let mut cfg = BroadcastConfig::new();
        cfg.configure_channel(0, 42);
        cfg.configure_channel(3, 55);
        cfg.configure_channel(7, 42);

        // Event 42 is active, 55 is not.
        let pending = cfg.pending_broadcasts(&|id| id == 42);
        assert_eq!(pending.len(), 2);
        assert!(pending.contains(&(0, 42)));
        assert!(pending.contains(&(7, 42)));
    }

    #[test]
    fn test_pending_broadcasts_disabled_channel() {
        let cfg = BroadcastConfig::new();
        // All channels disabled (event=0). Even if event 0 is "active",
        // channels with event=0 are considered disabled.
        let pending = cfg.pending_broadcasts(&|_| true);
        assert!(pending.is_empty());
    }

    #[test]
    fn test_broadcast_reset() {
        let mut cfg = BroadcastConfig::new();
        cfg.configure_channel(0, 42);
        cfg.block_channel(0, BroadcastDir::South);
        cfg.reset();

        assert_eq!(cfg.channels[0].event, 0);
        assert_eq!(cfg.read_block_value(BroadcastDir::South), 0);
    }

    #[test]
    fn test_broadcast_dir_from_index() {
        assert_eq!(BroadcastDir::from_index(0), Some(BroadcastDir::South));
        assert_eq!(BroadcastDir::from_index(1), Some(BroadcastDir::West));
        assert_eq!(BroadcastDir::from_index(2), Some(BroadcastDir::North));
        assert_eq!(BroadcastDir::from_index(3), Some(BroadcastDir::East));
        assert_eq!(BroadcastDir::from_index(4), None);
    }

    #[test]
    fn test_is_blocked_out_of_bounds() {
        let cfg = BroadcastConfig::new();
        // Out of bounds channel is always blocked.
        assert!(cfg.is_blocked(16, BroadcastDir::South));
    }
}
