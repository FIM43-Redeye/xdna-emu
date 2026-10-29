//! Legacy DMA stub types (buffer descriptors and channel state).

/// DMA buffer descriptor.
///
/// Describes a memory region for DMA transfer with multi-dimensional
/// addressing support.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct DmaBufferDescriptor {
    /// Base address (low 32 bits)
    pub addr_low: u32,
    /// Base address (high 32 bits, for 64-bit addressing)
    pub addr_high: u32,
    /// Transfer length in bytes
    pub length: u32,
    /// Control register (valid, compression, etc.)
    pub control: u32,
    /// Dimension 1 configuration (stride, wrap)
    pub d0: u32,
    /// Dimension 2 configuration
    pub d1: u32,
}

impl DmaBufferDescriptor {
    /// Check if this BD is valid (enabled)
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.control & 1 != 0
    }

    /// Get the base address as 64-bit
    #[inline]
    pub fn address(&self) -> u64 {
        ((self.addr_high as u64) << 32) | (self.addr_low as u64)
    }

    /// Get the next BD index (for chaining)
    #[inline]
    pub fn next_bd(&self) -> Option<u8> {
        let next = ((self.control >> 8) & 0xF) as u8;
        if self.control & 0x80 != 0 {
            // Use next BD bit set
            Some(next)
        } else {
            None
        }
    }
}

/// DMA channel state.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct DmaChannel {
    /// Control register
    pub control: u32,
    /// Start queue (BD to start)
    pub start_queue: u32,
    /// Current BD being processed
    pub current_bd: u8,
    /// Channel is running
    pub running: bool,
    /// Controller ID for task complete tokens (from control register bits 15:8)
    pub controller_id: u8,
    /// Finish-on-TLAST mode (S2MM only, from control register bits 17:16)
    pub fot_mode: u8,
    /// Enable token issue for current task (from start_queue bit 31)
    pub enable_token_issue: bool,
    /// Compression enable (MM2S only, from control register bit 4)
    pub compression_enable: bool,
    /// Decompression enable (S2MM only, from control register bit 4)
    pub decompression_enable: bool,
    /// Out-of-order mode enable (S2MM only, from control register bit 3)
    pub out_of_order_enable: bool,
    /// Status register (read-only bits updated during execution)
    pub status: u32,
}

impl DmaChannel {
    /// Check if channel is enabled
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.control & 1 != 0
    }

    /// Check if channel is paused
    #[inline]
    pub fn is_paused(&self) -> bool {
        self.control & 2 != 0
    }

    /// Check if channel is in reset
    #[inline]
    pub fn is_reset(&self) -> bool {
        (self.control >> 1) & 1 != 0
    }

    /// Get the controller ID for task complete tokens
    #[inline]
    pub fn get_controller_id(&self) -> u8 {
        self.controller_id
    }

    /// Get the FoT mode (S2MM only)
    #[inline]
    pub fn get_fot_mode(&self) -> u8 {
        self.fot_mode
    }

    /// Check if token issue is enabled for current task
    #[inline]
    pub fn should_issue_token(&self) -> bool {
        self.enable_token_issue
    }

    /// Update status register field: Cur_BD
    ///
    /// Uses the compute tile status layout. The DmaEngine.get_channel_status()
    /// method selects the correct layout per tile type; this is a convenience
    /// for the DmaChannel struct which stores a copy of the status word.
    pub fn set_cur_bd(&mut self, bd: u8) {
        let layout = &super::super::regdb::device_reg_layout().memory_status;
        self.status = layout.cur_bd.insert(self.status, bd as u32);
    }

    /// Update status register: Channel_Running
    pub fn set_channel_running(&mut self, running: bool) {
        let layout = &super::super::regdb::device_reg_layout().memory_status;
        if running {
            self.status = layout.channel_running.set_bit(self.status);
        } else {
            self.status &= !(layout.channel_running.mask << layout.channel_running.shift);
        }
    }

    /// Update status register: State bits (00=IDLE, 01=STARTING, 10=RUNNING)
    pub fn set_state(&mut self, state: u8) {
        let layout = &super::super::regdb::device_reg_layout().memory_status;
        self.status = layout.status.insert(self.status, state as u32);
    }
}
