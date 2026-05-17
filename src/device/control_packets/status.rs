//! Tile_Control_Packet_Handler_Status sticky-error conditions.
//!
//! Provides [`PktHandlerError`], the single place the
//! `Tile_Control_Packet_Handler_Status` bit positions are defined, so
//! detecting paths and the routing latch never spell raw bit literals.

/// A control-packet handler error with a faithful detecting path.
///
/// Bit positions per AM025 `Tile_Control_Packet_Handler_Status`
/// (regdb fields `First_Header_Parity_Error` /
/// `Second_Header_Parity_Error` / `Tlast_Error`). This is the single
/// source of truth for the bit map -- no other site names these
/// positions.
///
/// `SLVERR_On_Access` (bit `0x4`) is intentionally absent: it has no
/// faithful trigger until the successor plan repairs the runtime
/// register-access path. That plan adds the `Slverr` variant here; the
/// exhaustive no-`_` match below will then force every consumer to
/// handle it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PktHandlerError {
    /// Stream routing header (First) odd-parity failure -- bit 0.
    FirstHeaderParity,
    /// Control-packet opcode header (Second) odd-parity failure -- bit 1.
    SecondHeaderParity,
    /// TLAST in the wrong position on a write-class packet -- bit 3.
    Tlast,
}

impl PktHandlerError {
    /// The `Tile_Control_Packet_Handler_Status` bit this condition sets.
    pub fn bit(self) -> u32 {
        match self {
            PktHandlerError::FirstHeaderParity => 0x1,
            PktHandlerError::SecondHeaderParity => 0x2,
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
        assert_eq!(PktHandlerError::Tlast.bit(), 0x8);
    }
}
