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

#[derive(Default)]
pub(crate) struct PhoenixMailboxRegisters {
    words: BTreeMap<u32, u32>,
}

impl PhoenixMailboxRegisters {
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
