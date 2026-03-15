//! DMA transfer data carrier.
//!
//! A `Transfer` represents an active DMA operation from a single buffer
//! descriptor. It tracks the current position, handles multi-dimensional
//! addressing, and provides data queries for the channel FSM.
//!
//! State transitions (lock acquire/release, completion) are managed by the
//! `ChannelFsm` in `channel.rs` -- Transfer is a pure data carrier that
//! only advances its internal counters and address generator.
//!
//! # Data Flow
//!
//! For MM2S (Memory to Stream):
//! 1. Read data from tile memory at current address
//! 2. Send to stream switch
//! 3. Advance address generator
//!
//! For S2MM (Stream to Memory):
//! 1. Receive data from stream switch
//! 2. Write to tile memory at current address
//! 3. Advance address generator

mod padding;
mod core;

#[cfg(test)]
mod tests;

// Re-export all public types so external callers see no difference.
pub use self::padding::{PadAction, ZeroPadState};
pub use self::core::{
    LockAcquireMode,
    Transfer,
    TransferDirection,
    TransferEndpoint,
    parse_packet_type_from_header,
    parse_source_tile_from_header,
    parse_stream_id_from_header,
};
