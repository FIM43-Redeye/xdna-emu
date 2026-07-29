//! Phoenix BAR4 mailbox aperture shared by host and management firmware.
//!
//! Queue descriptors expose `0x030c_xxxx..0x030f_xxxx` device addresses to
//! the host. Firmware uses the corresponding `0x270c_xxxx..0x270f_xxxx`
//! addresses; both domains reach the same sparse register aperture.

use std::collections::BTreeMap;

// NPU1 `MPNPU_APERTURE2_BASE` through the end of its 0x0300_0000
// device-control envelope, per the pinned open driver.
const HOST_BASE: u32 = 0x030c_0000;
const FIRMWARE_BASE: u32 = 0x270c_0000;
const APERTURE_SIZE: u32 = 0x0004_0000;
const HOST_X2I_TAIL_BASE: u32 = 0x030d_0000;
const I2X_STATUS_OFFSET: u32 = 0x0001_1008;
const CHANNEL_STRIDE: u32 = 0x2000;
const CHANNEL_COUNT: u32 = 16;
const CONTROLLER_SOURCE_BASE: u8 = 0x20;

#[derive(Default)]
pub(crate) struct PhoenixMailboxRegisters {
    words: BTreeMap<u32, u32>,
}

impl PhoenixMailboxRegisters {
    /// Map an NPU1 host X2I-tail register to the source selected by the
    /// firmware's channel handler.
    pub(crate) fn host_x2i_source(address: u32) -> Option<u8> {
        let offset = address.checked_sub(HOST_X2I_TAIL_BASE)?;
        (offset < CHANNEL_COUNT * CHANNEL_STRIDE && offset % CHANNEL_STRIDE == 0)
            .then_some(CONTROLLER_SOURCE_BASE + (offset / CHANNEL_STRIDE) as u8)
    }

    /// Map either host or firmware I2X-status alias to its MSI-X channel.
    pub(crate) fn i2x_status_channel(address: u32) -> Option<u8> {
        let offset = Self::offset(address)?.checked_sub(I2X_STATUS_OFFSET)?;
        (offset < CHANNEL_COUNT * CHANNEL_STRIDE && offset % CHANNEL_STRIDE == 0)
            .then_some((offset / CHANNEL_STRIDE) as u8)
    }

    fn offset(address: u32) -> Option<u32> {
        [HOST_BASE, FIRMWARE_BASE]
            .into_iter()
            .find_map(|base| address.checked_sub(base).filter(|offset| *offset < APERTURE_SIZE))
            .filter(|offset| offset & 3 == 0)
    }

    pub(crate) fn read32(&self, address: u32) -> Option<u32> {
        Self::offset(address).map(|offset| self.words.get(&offset).copied().unwrap_or(0))
    }

    pub(crate) fn write32(&mut self, address: u32, value: u32) -> bool {
        let Some(offset) = Self::offset(address) else {
            return false;
        };
        self.words.insert(offset, value);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::PhoenixMailboxRegisters;

    #[test]
    fn i2x_status_channel_is_derived_for_every_host_and_firmware_alias() {
        for channel in 0..16u32 {
            let channel = channel as u8;
            assert_eq!(
                PhoenixMailboxRegisters::i2x_status_channel(0x030d_1008 + channel as u32 * 0x2000),
                Some(channel)
            );
            assert_eq!(
                PhoenixMailboxRegisters::i2x_status_channel(0x270d_1008 + channel as u32 * 0x2000),
                Some(channel)
            );
        }

        assert_eq!(PhoenixMailboxRegisters::i2x_status_channel(0x030d_1004), None);
        assert_eq!(PhoenixMailboxRegisters::i2x_status_channel(0x030d_100c), None);
        assert_eq!(PhoenixMailboxRegisters::i2x_status_channel(0x030f_1008), None);
    }
}
