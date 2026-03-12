//! Canonical signal identity for VCD deep extraction.
//!
//! Every VCD signal (from aiesimulator or from the emulator's own VCD output)
//! maps to a [`StatePath`] value. The comparison engine matches signals by
//! `StatePath` equality, so signals with the same canonical identity are
//! compared regardless of their VCD naming differences between sources.
//!
//! # Organisation
//!
//! Supporting types are defined first, roughly in dependency order:
//! - [`Subsystem`] -- top-level hardware subsystem classification
//! - [`DmaDir`] -- DMA direction (S2MM vs MM2S)
//! - [`PortBundle`] -- stream switch port bundle type
//! - [`PortId`] -- fully-qualified stream switch port identifier
//! - [`MemPortId`] -- memory port identifier
//!
//! [`StatePath`] is the main type. Its variants are grouped by subsystem and
//! each variant carries the minimum coordinates needed to uniquely identify a
//! hardware signal: (col, row) for the tile, plus subsystem-specific fields.

use std::fmt;

// ---------------------------------------------------------------------------
// Subsystem
// ---------------------------------------------------------------------------

/// Top-level hardware subsystem classification.
///
/// Used by the comparison engine to select the appropriate tolerance band and
/// by the coverage audit to group unmapped signals.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Subsystem {
    Core,
    Dma,
    Lock,
    Stream,
    Memory,
    Event,
    PerfCount,
}

impl Subsystem {
    /// All subsystem variants in `PartialOrd` order.
    pub const ALL: &'static [Subsystem] = &[
        Subsystem::Core,
        Subsystem::Dma,
        Subsystem::Lock,
        Subsystem::Stream,
        Subsystem::Memory,
        Subsystem::Event,
        Subsystem::PerfCount,
    ];

    /// Short lowercase string identifier for display and serialisation.
    pub fn as_str(self) -> &'static str {
        match self {
            Subsystem::Core => "core",
            Subsystem::Dma => "dma",
            Subsystem::Lock => "lock",
            Subsystem::Stream => "stream",
            Subsystem::Memory => "memory",
            Subsystem::Event => "event",
            Subsystem::PerfCount => "perf_count",
        }
    }
}

impl fmt::Display for Subsystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// DmaDir
// ---------------------------------------------------------------------------

/// DMA channel direction: stream-to-memory or memory-to-stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DmaDir {
    /// Stream-to-memory (inbound / receive).
    S2mm,
    /// Memory-to-stream (outbound / transmit).
    Mm2s,
}

impl DmaDir {
    /// Lowercase string representation matching aie-rt / aiesimulator convention.
    pub fn as_str(self) -> &'static str {
        match self {
            DmaDir::S2mm => "s2mm",
            DmaDir::Mm2s => "mm2s",
        }
    }
}

impl fmt::Display for DmaDir {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ---------------------------------------------------------------------------
// PortBundle
// ---------------------------------------------------------------------------

/// Stream switch port bundle type.
///
/// Bundle names follow the VCD naming convention used by aiesimulator, where
/// cardinal directions are lower-case (south, north, …) and the VCD prefix
/// carries the "s" prefix (sSouth, sDMA, …). See [`PortBundle::vcd_prefix`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PortBundle {
    North,
    South,
    East,
    West,
    Dma,
    Core,
    Fifo,
    Trace,
    TileCtrl,
}

impl PortBundle {
    /// The VCD signal prefix used by aiesimulator for ports in this bundle.
    ///
    /// Concatenate this prefix with a decimal index to form a port name that
    /// matches aiesimulator output, e.g. `"sSouth"` + `"3"` → `"sSouth3"`.
    pub fn vcd_prefix(self) -> &'static str {
        match self {
            PortBundle::North => "sNorth",
            PortBundle::South => "sSouth",
            PortBundle::East => "sEast",
            PortBundle::West => "sWest",
            PortBundle::Dma => "sDMA",
            PortBundle::Core => "sCore",
            PortBundle::Fifo => "sFIFO",
            PortBundle::Trace => "sTrace",
            PortBundle::TileCtrl => "sTileCtrl",
        }
    }
}

// ---------------------------------------------------------------------------
// PortId
// ---------------------------------------------------------------------------

/// Fully-qualified stream switch port identifier.
///
/// A port is identified by its VCD name string (e.g. `"sSouth3"`). Two
/// `PortId` values are equal if and only if their names are identical.
/// The `indexed` constructor builds the name from a [`PortBundle`] and an
/// integer index, producing the same string that aiesimulator would use.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PortId(String);

impl PortId {
    /// Construct a `PortId` from an explicit VCD name string.
    pub fn named(name: &str) -> Self {
        PortId(name.to_string())
    }

    /// Construct a `PortId` from a bundle type and an index.
    ///
    /// Produces the same string as `bundle.vcd_prefix()` concatenated with
    /// the decimal representation of `idx`.
    pub fn indexed(bundle: PortBundle, idx: u8) -> Self {
        PortId(format!("{}{}", bundle.vcd_prefix(), idx))
    }

    /// The underlying port name string.
    pub fn name(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for PortId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

// ---------------------------------------------------------------------------
// MemPortId
// ---------------------------------------------------------------------------

/// Memory module port identifier (used for memory bank access signals).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MemPortId {
    pub name: String,
}

impl fmt::Display for MemPortId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name)
    }
}

// ---------------------------------------------------------------------------
// StatePath
// ---------------------------------------------------------------------------

/// Canonical identity for a single VCD signal in the AIE2 hardware model.
///
/// Every VCD signal from aiesimulator (or from the emulator's VCD emission)
/// resolves to exactly one `StatePath`. The comparison engine uses `StatePath`
/// equality to pair signals across the two sources.
///
/// Variants are grouped by subsystem. Each variant carries (col, row) to
/// identify the tile, plus fields that uniquely identify the signal within
/// that tile's subsystem.
///
/// # Display format
///
/// `Display` produces a human-readable path of the form
/// `tile(col,row).subsystem.signal[index]`, e.g.:
/// - `tile(0,1).lock.value[3]`
/// - `tile(1,2).dma.s2mm[0].fsm_state`
/// - `tile(0,0).stream.sSouth3.data`
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum StatePath {
    // ------------------------------------------------------------------
    // Lock subsystem
    // ------------------------------------------------------------------

    /// Current numerical value of a lock register.
    LockValue { col: u8, row: u8, idx: u8 },
    /// Lock operation (acquire / release) in progress.
    LockOp { col: u8, row: u8, idx: u8 },

    // ------------------------------------------------------------------
    // DMA subsystem
    // ------------------------------------------------------------------

    /// DMA channel FSM state.
    DmaFsmState { col: u8, row: u8, dir: DmaDir, ch: u8 },
    /// Index of the BD currently being processed.
    DmaCurrentBd { col: u8, row: u8, dir: DmaDir, ch: u8 },
    /// Transfer length remaining in the current BD.
    DmaBdLength { col: u8, row: u8, dir: DmaDir, ch: u8 },
    /// Current DMA address pointer.
    DmaAddress { col: u8, row: u8, dir: DmaDir, ch: u8 },
    /// Data word on the DMA bus.
    DmaData { col: u8, row: u8, dir: DmaDir, ch: u8 },
    /// Words transferred on the stream side.
    DmaProcessedStream { col: u8, row: u8, dir: DmaDir, ch: u8 },
    /// Words transferred on the memory side.
    DmaProcessedMem { col: u8, row: u8, dir: DmaDir, ch: u8 },
    /// Lock ID to acquire before starting the BD.
    DmaLockAcqId { col: u8, row: u8, dir: DmaDir, ch: u8 },
    /// Value to acquire the lock at.
    DmaLockAcqValue { col: u8, row: u8, dir: DmaDir, ch: u8 },
    /// Value to release the lock to after the BD completes.
    DmaLockRelValue { col: u8, row: u8, dir: DmaDir, ch: u8 },
    /// DMA channel status flags.
    DmaStatus { col: u8, row: u8, dir: DmaDir, ch: u8 },
    /// Whether the BD descriptor entry is valid.
    DmaBdValid { col: u8, row: u8, dir: DmaDir, ch: u8 },
    /// Packet ID embedded in packet-mode transfers.
    DmaPacketId { col: u8, row: u8, dir: DmaDir, ch: u8 },
    /// Packet mode enable flag on the BD.
    DmaEnablePacket { col: u8, row: u8, dir: DmaDir, ch: u8 },
    /// Next BD index (linked-list chaining).
    DmaNextBd { col: u8, row: u8, dir: DmaDir, ch: u8 },
    /// Whether to follow the next-BD chain after this BD.
    DmaUseNextBd { col: u8, row: u8, dir: DmaDir, ch: u8 },
    /// Suppress TLAST at the end of the BD transfer.
    DmaTlastSuppress { col: u8, row: u8, dir: DmaDir, ch: u8 },
    /// Step size for the iteration dimension.
    DmaIterStepsize { col: u8, row: u8, dir: DmaDir, ch: u8 },
    /// Current iteration counter value.
    DmaIterCurrent { col: u8, row: u8, dir: DmaDir, ch: u8 },
    /// Iteration wrap (repeat count).
    DmaIterWrap { col: u8, row: u8, dir: DmaDir, ch: u8 },

    // ------------------------------------------------------------------
    // Stream subsystem
    // ------------------------------------------------------------------

    /// Data word present on a stream switch port.
    StreamPortData { col: u8, row: u8, port: PortId },
    /// Port is idle (no transfer in progress).
    StreamPortIdle { col: u8, row: u8, port: PortId },
    /// Port is actively transferring data.
    StreamPortRunning { col: u8, row: u8, port: PortId },
    /// Port is stalled (waiting for downstream consumer or upstream producer).
    StreamPortStalled { col: u8, row: u8, port: PortId },
    /// TLAST signal on the stream port (end-of-packet marker).
    StreamPortTlast { col: u8, row: u8, port: PortId },

    // ------------------------------------------------------------------
    // Core subsystem
    // ------------------------------------------------------------------

    /// Program counter at a specific pipeline stage.
    CorePc { col: u8, row: u8, stage: u8 },
    /// Program memory address bus (read side).
    CorePmAddress { col: u8, row: u8 },
    /// Program memory data bus (instruction word returned).
    CorePmData { col: u8, row: u8 },
    /// Data memory address bus (load/store address).
    CoreTmAddress { col: u8, row: u8 },
    /// Data memory read data.
    CoreTmReadData { col: u8, row: u8 },
    /// Data memory write data.
    CoreTmWriteData { col: u8, row: u8 },
    /// Data memory load in progress.
    CoreTmLoad { col: u8, row: u8 },
    /// Data memory store in progress.
    CoreTmStore { col: u8, row: u8 },
    /// Core reset signal.
    CoreReset { col: u8, row: u8 },
    /// Core is halted at a breakpoint.
    CoreBreakpointHalted { col: u8, row: u8 },

    // ------------------------------------------------------------------
    // Memory subsystem
    // ------------------------------------------------------------------

    /// Memory bank conflict detected (stall inserted).
    MemBankConflict { col: u8, row: u8, bank: u8 },
    /// Address that caused a bank conflict.
    MemConflictAddr { col: u8, row: u8, bank: u8 },
    /// Memory port access signal (read or write active).
    MemPortAccess { col: u8, row: u8, port: MemPortId },

    // ------------------------------------------------------------------
    // Event subsystem
    // ------------------------------------------------------------------

    /// A trace event fired at this tile with the given event code.
    EventTrace { col: u8, row: u8, event_code: u16, event_name: String },

    // ------------------------------------------------------------------
    // Performance counter subsystem
    // ------------------------------------------------------------------

    /// Performance counter value.
    PerfCounter { col: u8, row: u8, idx: u8 },
}

impl StatePath {
    /// Return the hardware subsystem this signal belongs to.
    pub fn subsystem(&self) -> Subsystem {
        match self {
            StatePath::LockValue { .. } | StatePath::LockOp { .. } => Subsystem::Lock,

            StatePath::DmaFsmState { .. }
            | StatePath::DmaCurrentBd { .. }
            | StatePath::DmaBdLength { .. }
            | StatePath::DmaAddress { .. }
            | StatePath::DmaData { .. }
            | StatePath::DmaProcessedStream { .. }
            | StatePath::DmaProcessedMem { .. }
            | StatePath::DmaLockAcqId { .. }
            | StatePath::DmaLockAcqValue { .. }
            | StatePath::DmaLockRelValue { .. }
            | StatePath::DmaStatus { .. }
            | StatePath::DmaBdValid { .. }
            | StatePath::DmaPacketId { .. }
            | StatePath::DmaEnablePacket { .. }
            | StatePath::DmaNextBd { .. }
            | StatePath::DmaUseNextBd { .. }
            | StatePath::DmaTlastSuppress { .. }
            | StatePath::DmaIterStepsize { .. }
            | StatePath::DmaIterCurrent { .. }
            | StatePath::DmaIterWrap { .. } => Subsystem::Dma,

            StatePath::StreamPortData { .. }
            | StatePath::StreamPortIdle { .. }
            | StatePath::StreamPortRunning { .. }
            | StatePath::StreamPortStalled { .. }
            | StatePath::StreamPortTlast { .. } => Subsystem::Stream,

            StatePath::CorePc { .. }
            | StatePath::CorePmAddress { .. }
            | StatePath::CorePmData { .. }
            | StatePath::CoreTmAddress { .. }
            | StatePath::CoreTmReadData { .. }
            | StatePath::CoreTmWriteData { .. }
            | StatePath::CoreTmLoad { .. }
            | StatePath::CoreTmStore { .. }
            | StatePath::CoreReset { .. }
            | StatePath::CoreBreakpointHalted { .. } => Subsystem::Core,

            StatePath::MemBankConflict { .. }
            | StatePath::MemConflictAddr { .. }
            | StatePath::MemPortAccess { .. } => Subsystem::Memory,

            StatePath::EventTrace { .. } => Subsystem::Event,

            StatePath::PerfCounter { .. } => Subsystem::PerfCount,
        }
    }

    /// Extract the (col, row) tile coordinates from any variant.
    pub fn tile(&self) -> (u8, u8) {
        match self {
            StatePath::LockValue { col, row, .. }
            | StatePath::LockOp { col, row, .. }
            | StatePath::DmaFsmState { col, row, .. }
            | StatePath::DmaCurrentBd { col, row, .. }
            | StatePath::DmaBdLength { col, row, .. }
            | StatePath::DmaAddress { col, row, .. }
            | StatePath::DmaData { col, row, .. }
            | StatePath::DmaProcessedStream { col, row, .. }
            | StatePath::DmaProcessedMem { col, row, .. }
            | StatePath::DmaLockAcqId { col, row, .. }
            | StatePath::DmaLockAcqValue { col, row, .. }
            | StatePath::DmaLockRelValue { col, row, .. }
            | StatePath::DmaStatus { col, row, .. }
            | StatePath::DmaBdValid { col, row, .. }
            | StatePath::DmaPacketId { col, row, .. }
            | StatePath::DmaEnablePacket { col, row, .. }
            | StatePath::DmaNextBd { col, row, .. }
            | StatePath::DmaUseNextBd { col, row, .. }
            | StatePath::DmaTlastSuppress { col, row, .. }
            | StatePath::DmaIterStepsize { col, row, .. }
            | StatePath::DmaIterCurrent { col, row, .. }
            | StatePath::DmaIterWrap { col, row, .. }
            | StatePath::StreamPortData { col, row, .. }
            | StatePath::StreamPortIdle { col, row, .. }
            | StatePath::StreamPortRunning { col, row, .. }
            | StatePath::StreamPortStalled { col, row, .. }
            | StatePath::StreamPortTlast { col, row, .. }
            | StatePath::CorePc { col, row, .. }
            | StatePath::CorePmAddress { col, row }
            | StatePath::CorePmData { col, row }
            | StatePath::CoreTmAddress { col, row }
            | StatePath::CoreTmReadData { col, row }
            | StatePath::CoreTmWriteData { col, row }
            | StatePath::CoreTmLoad { col, row }
            | StatePath::CoreTmStore { col, row }
            | StatePath::CoreReset { col, row }
            | StatePath::CoreBreakpointHalted { col, row }
            | StatePath::MemBankConflict { col, row, .. }
            | StatePath::MemConflictAddr { col, row, .. }
            | StatePath::MemPortAccess { col, row, .. }
            | StatePath::EventTrace { col, row, .. }
            | StatePath::PerfCounter { col, row, .. } => (*col, *row),
        }
    }

    /// Return the signal-specific field name for tolerance table lookup.
    ///
    /// This is the leaf signal name without tile or index context, used to
    /// look up per-field tolerance bands (e.g., `"fsm_state"` tolerates larger
    /// timing differences than `"data"`).
    pub fn field_name(&self) -> &'static str {
        match self {
            StatePath::LockValue { .. } => "value",
            StatePath::LockOp { .. } => "op",
            StatePath::DmaFsmState { .. } => "fsm_state",
            StatePath::DmaCurrentBd { .. } => "current_bd",
            StatePath::DmaBdLength { .. } => "bd_length",
            StatePath::DmaAddress { .. } => "address",
            StatePath::DmaData { .. } => "data",
            StatePath::DmaProcessedStream { .. } => "processed_stream",
            StatePath::DmaProcessedMem { .. } => "processed_mem",
            StatePath::DmaLockAcqId { .. } => "lock_acq_id",
            StatePath::DmaLockAcqValue { .. } => "lock_acq_value",
            StatePath::DmaLockRelValue { .. } => "lock_rel_value",
            StatePath::DmaStatus { .. } => "status",
            StatePath::DmaBdValid { .. } => "bd_valid",
            StatePath::DmaPacketId { .. } => "packet_id",
            StatePath::DmaEnablePacket { .. } => "enable_packet",
            StatePath::DmaNextBd { .. } => "next_bd",
            StatePath::DmaUseNextBd { .. } => "use_next_bd",
            StatePath::DmaTlastSuppress { .. } => "tlast_suppress",
            StatePath::DmaIterStepsize { .. } => "iter_stepsize",
            StatePath::DmaIterCurrent { .. } => "iter_current",
            StatePath::DmaIterWrap { .. } => "iter_wrap",
            StatePath::StreamPortData { .. } => "data",
            StatePath::StreamPortIdle { .. } => "idle",
            StatePath::StreamPortRunning { .. } => "running",
            StatePath::StreamPortStalled { .. } => "stalled",
            StatePath::StreamPortTlast { .. } => "tlast",
            StatePath::CorePc { .. } => "pc",
            StatePath::CorePmAddress { .. } => "pm_address",
            StatePath::CorePmData { .. } => "pm_data",
            StatePath::CoreTmAddress { .. } => "tm_address",
            StatePath::CoreTmReadData { .. } => "tm_read_data",
            StatePath::CoreTmWriteData { .. } => "tm_write_data",
            StatePath::CoreTmLoad { .. } => "tm_load",
            StatePath::CoreTmStore { .. } => "tm_store",
            StatePath::CoreReset { .. } => "reset",
            StatePath::CoreBreakpointHalted { .. } => "breakpoint_halted",
            StatePath::MemBankConflict { .. } => "bank_conflict",
            StatePath::MemConflictAddr { .. } => "conflict_addr",
            StatePath::MemPortAccess { .. } => "port_access",
            StatePath::EventTrace { .. } => "event",
            StatePath::PerfCounter { .. } => "counter",
        }
    }
}

impl fmt::Display for StatePath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (col, row) = self.tile();
        let sub = self.subsystem();
        match self {
            // Lock
            StatePath::LockValue { idx, .. } => {
                write!(f, "tile({},{}).{}.value[{}]", col, row, sub, idx)
            }
            StatePath::LockOp { idx, .. } => {
                write!(f, "tile({},{}).{}.op[{}]", col, row, sub, idx)
            }

            // DMA
            StatePath::DmaFsmState { dir, ch, .. } => {
                write!(f, "tile({},{}).{}.{}[{}].fsm_state", col, row, sub, dir, ch)
            }
            StatePath::DmaCurrentBd { dir, ch, .. } => {
                write!(f, "tile({},{}).{}.{}[{}].current_bd", col, row, sub, dir, ch)
            }
            StatePath::DmaBdLength { dir, ch, .. } => {
                write!(f, "tile({},{}).{}.{}[{}].bd_length", col, row, sub, dir, ch)
            }
            StatePath::DmaAddress { dir, ch, .. } => {
                write!(f, "tile({},{}).{}.{}[{}].address", col, row, sub, dir, ch)
            }
            StatePath::DmaData { dir, ch, .. } => {
                write!(f, "tile({},{}).{}.{}[{}].data", col, row, sub, dir, ch)
            }
            StatePath::DmaProcessedStream { dir, ch, .. } => {
                write!(f, "tile({},{}).{}.{}[{}].processed_stream", col, row, sub, dir, ch)
            }
            StatePath::DmaProcessedMem { dir, ch, .. } => {
                write!(f, "tile({},{}).{}.{}[{}].processed_mem", col, row, sub, dir, ch)
            }
            StatePath::DmaLockAcqId { dir, ch, .. } => {
                write!(f, "tile({},{}).{}.{}[{}].lock_acq_id", col, row, sub, dir, ch)
            }
            StatePath::DmaLockAcqValue { dir, ch, .. } => {
                write!(f, "tile({},{}).{}.{}[{}].lock_acq_value", col, row, sub, dir, ch)
            }
            StatePath::DmaLockRelValue { dir, ch, .. } => {
                write!(f, "tile({},{}).{}.{}[{}].lock_rel_value", col, row, sub, dir, ch)
            }
            StatePath::DmaStatus { dir, ch, .. } => {
                write!(f, "tile({},{}).{}.{}[{}].status", col, row, sub, dir, ch)
            }
            StatePath::DmaBdValid { dir, ch, .. } => {
                write!(f, "tile({},{}).{}.{}[{}].bd_valid", col, row, sub, dir, ch)
            }
            StatePath::DmaPacketId { dir, ch, .. } => {
                write!(f, "tile({},{}).{}.{}[{}].packet_id", col, row, sub, dir, ch)
            }
            StatePath::DmaEnablePacket { dir, ch, .. } => {
                write!(f, "tile({},{}).{}.{}[{}].enable_packet", col, row, sub, dir, ch)
            }
            StatePath::DmaNextBd { dir, ch, .. } => {
                write!(f, "tile({},{}).{}.{}[{}].next_bd", col, row, sub, dir, ch)
            }
            StatePath::DmaUseNextBd { dir, ch, .. } => {
                write!(f, "tile({},{}).{}.{}[{}].use_next_bd", col, row, sub, dir, ch)
            }
            StatePath::DmaTlastSuppress { dir, ch, .. } => {
                write!(f, "tile({},{}).{}.{}[{}].tlast_suppress", col, row, sub, dir, ch)
            }
            StatePath::DmaIterStepsize { dir, ch, .. } => {
                write!(f, "tile({},{}).{}.{}[{}].iter_stepsize", col, row, sub, dir, ch)
            }
            StatePath::DmaIterCurrent { dir, ch, .. } => {
                write!(f, "tile({},{}).{}.{}[{}].iter_current", col, row, sub, dir, ch)
            }
            StatePath::DmaIterWrap { dir, ch, .. } => {
                write!(f, "tile({},{}).{}.{}[{}].iter_wrap", col, row, sub, dir, ch)
            }

            // Stream
            StatePath::StreamPortData { port, .. } => {
                write!(f, "tile({},{}).{}.{}.data", col, row, sub, port)
            }
            StatePath::StreamPortIdle { port, .. } => {
                write!(f, "tile({},{}).{}.{}.idle", col, row, sub, port)
            }
            StatePath::StreamPortRunning { port, .. } => {
                write!(f, "tile({},{}).{}.{}.running", col, row, sub, port)
            }
            StatePath::StreamPortStalled { port, .. } => {
                write!(f, "tile({},{}).{}.{}.stalled", col, row, sub, port)
            }
            StatePath::StreamPortTlast { port, .. } => {
                write!(f, "tile({},{}).{}.{}.tlast", col, row, sub, port)
            }

            // Core
            StatePath::CorePc { stage, .. } => {
                write!(f, "tile({},{}).{}.pc[{}]", col, row, sub, stage)
            }
            StatePath::CorePmAddress { .. } => {
                write!(f, "tile({},{}).{}.pm_address", col, row, sub)
            }
            StatePath::CorePmData { .. } => {
                write!(f, "tile({},{}).{}.pm_data", col, row, sub)
            }
            StatePath::CoreTmAddress { .. } => {
                write!(f, "tile({},{}).{}.tm_address", col, row, sub)
            }
            StatePath::CoreTmReadData { .. } => {
                write!(f, "tile({},{}).{}.tm_read_data", col, row, sub)
            }
            StatePath::CoreTmWriteData { .. } => {
                write!(f, "tile({},{}).{}.tm_write_data", col, row, sub)
            }
            StatePath::CoreTmLoad { .. } => {
                write!(f, "tile({},{}).{}.tm_load", col, row, sub)
            }
            StatePath::CoreTmStore { .. } => {
                write!(f, "tile({},{}).{}.tm_store", col, row, sub)
            }
            StatePath::CoreReset { .. } => {
                write!(f, "tile({},{}).{}.reset", col, row, sub)
            }
            StatePath::CoreBreakpointHalted { .. } => {
                write!(f, "tile({},{}).{}.breakpoint_halted", col, row, sub)
            }

            // Memory
            StatePath::MemBankConflict { bank, .. } => {
                write!(f, "tile({},{}).{}.bank_conflict[{}]", col, row, sub, bank)
            }
            StatePath::MemConflictAddr { bank, .. } => {
                write!(f, "tile({},{}).{}.conflict_addr[{}]", col, row, sub, bank)
            }
            StatePath::MemPortAccess { port, .. } => {
                write!(f, "tile({},{}).{}.{}.access", col, row, sub, port)
            }

            // Events
            StatePath::EventTrace { event_code, event_name, .. } => {
                write!(f, "tile({},{}).{}.{}[{:#06x}]", col, row, sub, event_name, event_code)
            }

            // Perf counters
            StatePath::PerfCounter { idx, .. } => {
                write!(f, "tile({},{}).{}.counter[{}]", col, row, sub, idx)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_path_subsystem_classification() {
        let lock = StatePath::LockValue { col: 0, row: 1, idx: 3 };
        assert_eq!(lock.subsystem(), Subsystem::Lock);
        let dma = StatePath::DmaFsmState { col: 1, row: 2, dir: DmaDir::S2mm, ch: 0 };
        assert_eq!(dma.subsystem(), Subsystem::Dma);
        let stream = StatePath::StreamPortData { col: 0, row: 0, port: PortId::named("sSouth3") };
        assert_eq!(stream.subsystem(), Subsystem::Stream);
        let core = StatePath::CorePc { col: 0, row: 3, stage: 1 };
        assert_eq!(core.subsystem(), Subsystem::Core);
    }

    #[test]
    fn state_path_display_roundtrip() {
        let path = StatePath::LockValue { col: 2, row: 1, idx: 7 };
        let s = path.to_string();
        assert!(s.contains("lock"));
        assert!(s.contains("2"));
        assert!(s.contains("1"));
        assert!(s.contains("7"));
    }

    #[test]
    fn state_path_equality_and_hash() {
        use std::collections::HashSet;
        let a = StatePath::LockValue { col: 0, row: 1, idx: 3 };
        let b = StatePath::LockValue { col: 0, row: 1, idx: 3 };
        let c = StatePath::LockValue { col: 0, row: 1, idx: 4 };
        assert_eq!(a, b);
        assert_ne!(a, c);
        let mut set = HashSet::new();
        set.insert(a);
        assert!(set.contains(&b));
        assert!(!set.contains(&c));
    }

    #[test]
    fn port_id_construction() {
        let named = PortId::named("sSouth3");
        assert_eq!(named.name(), "sSouth3");
        let indexed = PortId::indexed(PortBundle::South, 3);
        assert_eq!(indexed.name(), "sSouth3");
        assert_eq!(named, indexed);
    }

    #[test]
    fn dma_dir_display() {
        assert_eq!(DmaDir::S2mm.as_str(), "s2mm");
        assert_eq!(DmaDir::Mm2s.as_str(), "mm2s");
    }

    // Additional coverage tests beyond the required set

    #[test]
    fn subsystem_display() {
        assert_eq!(Subsystem::Lock.to_string(), "lock");
        assert_eq!(Subsystem::Dma.to_string(), "dma");
        assert_eq!(Subsystem::Core.to_string(), "core");
        assert_eq!(Subsystem::Stream.to_string(), "stream");
        assert_eq!(Subsystem::Memory.to_string(), "memory");
        assert_eq!(Subsystem::Event.to_string(), "event");
        assert_eq!(Subsystem::PerfCount.to_string(), "perf_count");
    }

    #[test]
    fn subsystem_all_covers_all_variants() {
        // Every subsystem variant must appear in ALL exactly once.
        let all = Subsystem::ALL;
        assert_eq!(all.len(), 7);
        for &s in all {
            assert_eq!(all.iter().filter(|&&x| x == s).count(), 1);
        }
    }

    #[test]
    fn port_bundle_vcd_prefix_spot_check() {
        assert_eq!(PortBundle::Dma.vcd_prefix(), "sDMA");
        assert_eq!(PortBundle::North.vcd_prefix(), "sNorth");
        assert_eq!(PortBundle::TileCtrl.vcd_prefix(), "sTileCtrl");
    }

    #[test]
    fn state_path_tile_extraction() {
        let p = StatePath::DmaFsmState { col: 3, row: 5, dir: DmaDir::Mm2s, ch: 1 };
        assert_eq!(p.tile(), (3, 5));
        let p2 = StatePath::CorePc { col: 1, row: 2, stage: 0 };
        assert_eq!(p2.tile(), (1, 2));
    }

    #[test]
    fn state_path_field_name() {
        assert_eq!(StatePath::LockValue { col: 0, row: 0, idx: 0 }.field_name(), "value");
        assert_eq!(StatePath::DmaFsmState { col: 0, row: 0, dir: DmaDir::S2mm, ch: 0 }.field_name(), "fsm_state");
        assert_eq!(StatePath::CorePc { col: 0, row: 0, stage: 0 }.field_name(), "pc");
        assert_eq!(StatePath::PerfCounter { col: 0, row: 0, idx: 0 }.field_name(), "counter");
    }

    #[test]
    fn dma_dir_display_trait() {
        assert_eq!(format!("{}", DmaDir::S2mm), "s2mm");
        assert_eq!(format!("{}", DmaDir::Mm2s), "mm2s");
    }

    #[test]
    fn event_trace_display() {
        let p = StatePath::EventTrace {
            col: 0,
            row: 2,
            event_code: 0x0042,
            event_name: "core_active".to_string(),
        };
        let s = p.to_string();
        assert!(s.contains("tile(0,2)"));
        assert!(s.contains("event"));
        assert!(s.contains("core_active"));
    }

    #[test]
    fn mem_port_access_display() {
        let p = StatePath::MemPortAccess {
            col: 1,
            row: 3,
            port: MemPortId { name: "portA".to_string() },
        };
        let s = p.to_string();
        assert!(s.contains("memory"));
        assert!(s.contains("portA"));
    }

    #[test]
    fn stream_port_data_display() {
        let p = StatePath::StreamPortData {
            col: 0,
            row: 1,
            port: PortId::indexed(PortBundle::North, 0),
        };
        let s = p.to_string();
        assert!(s.contains("stream"));
        assert!(s.contains("sNorth0"));
        assert!(s.contains("data"));
    }
}
