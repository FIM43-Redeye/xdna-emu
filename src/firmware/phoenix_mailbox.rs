//! Phoenix BAR4 mailbox words published by the management-channel descriptor.

#[derive(Default)]
pub(crate) struct PhoenixMailboxRegisters {
    x2i_tail: u32,
    x2i_head: u32,
    i2x_tail: u32,
    i2x_head: u32,
    i2x_status: u32,
}

impl PhoenixMailboxRegisters {
    pub(crate) fn read32(&self, address: u32) -> Option<u32> {
        match address {
            0x030e_c000 => Some(self.x2i_tail),
            0x030e_c004 => Some(self.x2i_head),
            0x030e_d000 => Some(self.i2x_tail),
            0x030e_d004 => Some(self.i2x_head),
            0x030e_d008 => Some(self.i2x_status),
            _ => None,
        }
    }

    pub(crate) fn write32(&mut self, address: u32, value: u32) -> bool {
        match address {
            0x030e_c000 => self.x2i_tail = value,
            0x030e_c004 => self.x2i_head = value,
            0x030e_d000 => self.i2x_tail = value,
            0x030e_d004 => self.i2x_head = value,
            0x030e_d008 => self.i2x_status = value,
            _ => return false,
        }
        true
    }
}
