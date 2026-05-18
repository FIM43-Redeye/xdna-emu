//! Tile_Control_Packet_Handler_Status sticky-error conditions.
//!
//! Provides [`PktHandlerError`], the single place the
//! `Tile_Control_Packet_Handler_Status` bit positions are defined, so
//! detecting paths and the routing latch never spell raw bit literals.

/// A control-packet handler error with a faithful detecting path.
///
/// Bit positions per AM025 `Tile_Control_Packet_Handler_Status` and
/// aie-rt `XAIEGBL_CORE_VALUE_TILCTRLPKTHANSTA`
/// (`xaiegbl_params.h:7761`: `(Tlast<<3)+(SLVERR<<2)+(Second<<1)+
/// (First<<0)`). This is the single source of truth for the bit map --
/// no other site names these positions.
///
/// `Slverr` is the AXI slave-error (DECERR) analogue: a control-packet
/// access whose offset decodes to no slave (AM025
/// `aie_aximm_config.txt:2-3`, `SLVERR_Block` -- "SLVERR when accessing
/// unmapped registers"). All four are poll-only sticky bits: the
/// handler latches and continues; firmware polls. The `SLVERR_Block`
/// config-suppression refinement is a tracked NoC-gated goal (spec
/// Section 10), not modeled here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PktHandlerError {
    /// Stream routing header (First) odd-parity failure -- bit 0.
    FirstHeaderParity,
    /// Control-packet opcode header (Second) odd-parity failure -- bit 1.
    SecondHeaderParity,
    /// Register access to an undecoded offset (AXI SLVERR) -- bit 2.
    Slverr,
    /// TLAST in the wrong position on a write-class packet -- bit 3.
    Tlast,
}

impl PktHandlerError {
    /// The `Tile_Control_Packet_Handler_Status` bit this condition sets.
    pub fn bit(self) -> u32 {
        match self {
            PktHandlerError::FirstHeaderParity => 0x1,
            PktHandlerError::SecondHeaderParity => 0x2,
            PktHandlerError::Slverr => 0x4,
            PktHandlerError::Tlast => 0x8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bit_map_matches_am025() {
        assert_eq!(PktHandlerError::FirstHeaderParity.bit(), 0x1);
        assert_eq!(PktHandlerError::SecondHeaderParity.bit(), 0x2);
        assert_eq!(PktHandlerError::Slverr.bit(), 0x4);
        assert_eq!(PktHandlerError::Tlast.bit(), 0x8);
    }
}
