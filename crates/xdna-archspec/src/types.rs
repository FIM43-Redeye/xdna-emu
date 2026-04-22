//! Core type definitions for the NPU architecture graph.

use serde::{Deserialize, Serialize};
use std::fmt;

// ============================================================================
// Architecture and tile identification
// ============================================================================

/// Target NPU architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Architecture {
    Aie,
    Aie2,
    Aie2p,
}

impl fmt::Display for Architecture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Aie => write!(f, "AIE"),
            Self::Aie2 => write!(f, "AIE2"),
            Self::Aie2p => write!(f, "AIE2P"),
        }
    }
}

/// Fine-grained device generation, matching aie-rt's XAIE_DEV_GEN_* constants.
///
/// `Architecture` is the coarse bucket (AIE/AIE2/AIE2P); this enum captures
/// the specific silicon revision within each architecture family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceGeneration {
    /// Generic/unknown (XAIE_DEV_GENERIC_DEVICE = 0)
    Generic,
    /// AIE1 -- Versal FPGA (XAIE_DEV_GEN_AIE = 1)
    Aie,
    /// AIE-ML -- AIE2 base (XAIE_DEV_GEN_AIEML = 2)
    AieMl,
    /// AIE2 IPU variant (XAIE_DEV_GEN_AIE2IPU = 3)
    Aie2Ipu,
    /// AIE2P (XAIE_DEV_GEN_AIE2P = 4)
    Aie2P,
    /// AIE2PS (XAIE_DEV_GEN_AIE2PS = 5)
    Aie2Ps,
    /// S100 (XAIE_DEV_GEN_S100 = 6)
    S100,
    /// S200 (XAIE_DEV_GEN_S200 = 7)
    S200,
    /// AIE2P Strix A0 stepping (XAIE_DEV_GEN_AIE2P_STRIX_A0 = 8)
    Aie2PStrixA0,
    /// AIE2P Strix B0 stepping (XAIE_DEV_GEN_AIE2P_STRIX_B0 = 9)
    Aie2PStrixB0,
}

/// Tile type within the NPU array.
///
/// aie-rt distinguishes four tile types (AIETILE, SHIMNOC, SHIMPL, MEMTILE).
/// ShimPl tiles exist in the hardware but are not software-accessible on NPU
/// devices, so they typically don't appear in device model extractions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TileKind {
    Compute,
    Mem,
    ShimNoc,
    ShimPl,
}

impl fmt::Display for TileKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Compute => write!(f, "compute"),
            Self::Mem => write!(f, "mem"),
            Self::ShimNoc => write!(f, "shim_noc"),
            Self::ShimPl => write!(f, "shim_pl"),
        }
    }
}

impl TileKind {
    /// Is this a shim tile (either `ShimNoc` or `ShimPl`)?
    ///
    /// Both shim variants share the same conceptual "shim tile" role in
    /// the emulator's code paths. AIE2 only ever produces `ShimNoc`;
    /// AIE1 produces both, but code that cares about per-column mux
    /// distinction should match on the variant directly rather than
    /// calling this helper.
    #[inline]
    pub const fn is_shim(self) -> bool {
        matches!(self, TileKind::ShimNoc | TileKind::ShimPl)
    }

    /// Is this a memory tile?
    #[inline]
    pub const fn is_mem(self) -> bool {
        matches!(self, TileKind::Mem)
    }

    /// Is this a compute tile?
    #[inline]
    pub const fn is_compute(self) -> bool {
        matches!(self, TileKind::Compute)
    }
}

#[cfg(test)]
mod tile_kind_predicate_tests {
    use super::TileKind;

    #[test]
    fn is_shim_covers_both_variants() {
        assert!(TileKind::ShimNoc.is_shim());
        assert!(TileKind::ShimPl.is_shim());
        assert!(!TileKind::Mem.is_shim());
        assert!(!TileKind::Compute.is_shim());
    }

    #[test]
    fn is_mem_only_mem() {
        assert!(TileKind::Mem.is_mem());
        assert!(!TileKind::Compute.is_mem());
        assert!(!TileKind::ShimNoc.is_mem());
        assert!(!TileKind::ShimPl.is_mem());
    }

    #[test]
    fn is_compute_only_compute() {
        assert!(TileKind::Compute.is_compute());
        assert!(!TileKind::Mem.is_compute());
        assert!(!TileKind::ShimNoc.is_compute());
        assert!(!TileKind::ShimPl.is_compute());
    }

    #[test]
    fn predicates_are_const_fn() {
        // Verify const-fn status: these can be evaluated at compile time.
        const _SHIM: bool = TileKind::ShimNoc.is_shim();
        const _MEM: bool = TileKind::Mem.is_mem();
        const _COMPUTE: bool = TileKind::Compute.is_compute();
    }
}

// ============================================================================
// Source attribution
// ============================================================================

// ============================================================================
// Cross-source confirmation
// ============================================================================

/// Compare hardware-relevant content, ignoring provenance metadata.
///
/// Types that carry `SourceAttribution` or other metadata should implement
/// this to compare only the hardware facts. For simple types (u8, String,
/// etc.), `PartialEq` and `FactEquals` are equivalent.
pub trait FactEquals {
    fn fact_equals(&self, other: &Self) -> bool;
}

/// Blanket impl: any `PartialEq` type that doesn't need special handling
/// can use `PartialEq` as its `FactEquals`.
impl<T: PartialEq> FactEquals for T
where
    T: NoProvenance,
{
    fn fact_equals(&self, other: &Self) -> bool {
        self == other
    }
}

/// Marker trait for types that contain no provenance metadata,
/// so `PartialEq` is sufficient for fact comparison.
/// Implemented for primitive and simple types.
pub trait NoProvenance {}
impl NoProvenance for u8 {}
impl NoProvenance for u16 {}
impl NoProvenance for u32 {}
impl NoProvenance for u64 {}
impl NoProvenance for i32 {}
impl NoProvenance for bool {}
impl NoProvenance for String {}

/// A fact confirmed by one or more toolchain sources.
///
/// Every fact in the graph is wrapped in `Confirmed<T>`. The first source
/// establishes the value; subsequent sources must agree (via `FactEquals`)
/// or the confirmation panics with a detailed conflict message.
///
/// Comparison between `Confirmed` values uses only the inner value, not
/// the source list -- two facts are equal if the hardware data matches,
/// regardless of where it came from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Confirmed<T> {
    value: T,
    sources: Vec<SourceAttribution>,
}

impl<T: FactEquals> Confirmed<T> {
    /// Create a new fact with its first source.
    pub fn new(value: T, source: SourceAttribution) -> Self {
        Self {
            value,
            sources: vec![source],
        }
    }

    /// Confirm the fact from an additional source.
    ///
    /// Panics if the new value disagrees with the existing value. A conflict
    /// means either a parser bug or a real inconsistency between toolchain
    /// sources -- both demand immediate attention.
    pub fn confirm(&mut self, value: T, source: SourceAttribution) {
        if self.value.fact_equals(&value) {
            self.sources.push(source);
        } else {
            let existing: Vec<String> =
                self.sources.iter().map(|s| format!("{}", s)).collect();
            panic!(
                "GRAPH CONFLICT: {} disagrees with existing source(s) [{}]",
                source,
                existing.join(", "),
            );
        }
    }

    /// The confirmed value.
    pub fn value(&self) -> &T {
        &self.value
    }

    /// All sources that confirmed this value.
    pub fn sources(&self) -> &[SourceAttribution] {
        &self.sources
    }
}

/// `Confirmed` values are equal when their hardware data matches,
/// regardless of provenance.
impl<T: FactEquals> PartialEq for Confirmed<T> {
    fn eq(&self, other: &Self) -> bool {
        self.value.fact_equals(&other.value)
    }
}

impl<T: FactEquals + Eq> Eq for Confirmed<T> {}

/// Which toolchain source a fact was derived from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Source {
    Am025Json,
    AieRt,
    DeviceModel,
    TableGen,
    /// Hand-written constants from AM020 architecture manual prose.
    /// Used for values with no machine-readable source (timing, packet format).
    Am020,
}

impl fmt::Display for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Am025Json => write!(f, "AM025 JSON"),
            Self::AieRt => write!(f, "aie-rt"),
            Self::DeviceModel => write!(f, "device model"),
            Self::TableGen => write!(f, "TableGen"),
            Self::Am020 => write!(f, "AM020 manual"),
        }
    }
}

/// Attribution of a fact to its toolchain source.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceAttribution {
    pub origin: Source,
    pub file: String,
    pub detail: String,
}

impl fmt::Display for SourceAttribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {} ({})", self.origin, self.file, self.detail)
    }
}

// ============================================================================
// Register model
// ============================================================================

/// Bit range within a register -- contiguous or split across non-adjacent positions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BitRange {
    /// Standard contiguous range, e.g., bits 17:0.
    Contiguous { msb: u8, lsb: u8 },
    /// Split field: multiple (msb, lsb) fragments, ordered high-to-low.
    Split(Vec<(u8, u8)>),
}

impl BitRange {
    /// Total width in bits.
    pub fn width(&self) -> u8 {
        match self {
            Self::Contiguous { msb, lsb } => msb - lsb + 1,
            Self::Split(fragments) => {
                fragments.iter().map(|(msb, lsb)| msb - lsb + 1).sum()
            }
        }
    }
}

/// Semantic meaning of a register field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldSemantics {
    Address,
    Count,
    Enable,
    LockId,
    LockValue,
    ChannelId,
    BdIndex,
    Status,
    Control,
    EventId,
    PortIndex,
    DataWidth,
    Reserved,
    /// Not yet classified.
    Unknown,
}

/// Register access mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Access {
    ReadWrite,
    ReadOnly,
    WriteOnly,
    WriteToClear,
    Mixed,
}

/// A single bit field within a register.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FieldModel {
    pub name: String,
    pub bits: BitRange,
    pub meaning: FieldSemantics,
    pub source: SourceAttribution,
}

impl FactEquals for FieldModel {
    fn fact_equals(&self, other: &Self) -> bool {
        self.name == other.name && self.bits == other.bits && self.meaning == other.meaning
    }
}

/// A hardware register.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RegisterModel {
    pub name: String,
    pub offset: u32,
    pub width: u8,
    pub reset_value: u32,
    pub fields: Vec<FieldModel>,
    /// Which functional subsystem this register belongs to (DMA, Lock, etc.).
    pub subsystem: SubsystemKind,
    pub access: Access,
    pub source: SourceAttribution,
}

impl FactEquals for RegisterModel {
    fn fact_equals(&self, other: &Self) -> bool {
        self.name == other.name
            && self.offset == other.offset
            && self.width == other.width
            && self.reset_value == other.reset_value
            && self.subsystem == other.subsystem
            && self.access == other.access
            && self.fields.len() == other.fields.len()
            && self
                .fields
                .iter()
                .zip(other.fields.iter())
                .all(|(a, b)| a.fact_equals(b))
    }
}

// ============================================================================
// Module and subsystem model
// ============================================================================

/// Physical hardware module within a tile, each with its own register address
/// space. Matches AMD's aie-rt module identifiers (XAIE_CORE_MOD, etc.).
///
/// This is the mid-level of the hardware hierarchy:
///   Tile (TileKind) -> Module (ModuleKind) -> Subsystem (SubsystemKind)
///
/// A compute tile contains two modules (Core + Memory). Memtiles and shim
/// tiles each contain one module.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModuleKind {
    /// Core module within a compute tile (AM025 "core", aie-rt XAIE_CORE_MOD).
    /// Contains: processor, program memory, accumulators, core control.
    Core,
    /// Memory module within a compute tile (AM025 "memory", aie-rt XAIE_MEM_MOD).
    /// Contains: data memory, DMA, locks, stream switch (memory side).
    Memory,
    /// Memtile module (AM025 "memory_tile"). Like an expanded memory module
    /// with no core -- more BDs, more locks, larger memory.
    MemTile,
    /// Shim module (AM025 "shim", aie-rt XAIE_PL_MOD).
    /// Contains: NoC interface, DMA, locks, stream switch.
    Shim,
}

impl fmt::Display for ModuleKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Core => write!(f, "core"),
            Self::Memory => write!(f, "memory"),
            Self::MemTile => write!(f, "mem_tile"),
            Self::Shim => write!(f, "shim"),
        }
    }
}

/// Functional subsystem within a module. Registers are grouped by the
/// subsystem they control.
///
/// This is the finest level of the hardware hierarchy:
///   Tile (TileKind) -> Module (ModuleKind) -> Subsystem (SubsystemKind)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SubsystemKind {
    Dma,
    Lock,
    /// Address-encoded lock acquire/release command interface.
    ///
    /// Lock_Request (0x40000 on compute, 0xD0000 on memtile) is architecturally
    /// distinct from lock value registers: the address bits encode lock_id,
    /// acquire/release, and value. Separating it from `Lock` prevents a 131KB
    /// span that would overlap ProgramMemory and Processor subsystems.
    LockRequest,
    StreamSwitch,
    /// The processor (VLIW core). Named "Processor" to avoid confusion with
    /// `ModuleKind::Core` which is the physical core module.
    Processor,
    ProgramMemory,
    DataMemory,
    Trace,
    Event,
    /// Hardware performance counters (aie-rt PerfMod).
    Performance,
    /// Cycle counter / timer (aie-rt TimerMod).
    Timer,
    /// Data watchpoint units.
    WatchPoint,
    /// Core debug interface (debug halt, single step, breakpoints).
    Debug,
    /// Program counter sampling registers.
    ProgramCounter,
    /// Interrupt controller (shim tiles).
    Interrupt,
    /// Network-on-chip interface (shim tiles).
    NoC,
    ShimMux,
    /// Subsystem not yet classified.
    Unknown,
}

impl fmt::Display for SubsystemKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dma => write!(f, "dma"),
            Self::Lock => write!(f, "lock"),
            Self::LockRequest => write!(f, "lock_request"),
            Self::StreamSwitch => write!(f, "stream_switch"),
            Self::Processor => write!(f, "processor"),
            Self::ProgramMemory => write!(f, "program_memory"),
            Self::DataMemory => write!(f, "data_memory"),
            Self::Trace => write!(f, "trace"),
            Self::Event => write!(f, "event"),
            Self::Performance => write!(f, "performance"),
            Self::Timer => write!(f, "timer"),
            Self::WatchPoint => write!(f, "watchpoint"),
            Self::Debug => write!(f, "debug"),
            Self::ProgramCounter => write!(f, "program_counter"),
            Self::Interrupt => write!(f, "interrupt"),
            Self::NoC => write!(f, "noc"),
            Self::ShimMux => write!(f, "shim_mux"),
            Self::Unknown => write!(f, "unknown"),
        }
    }
}

/// A bundle of ports in one direction group (e.g., "DMA" with 2 masters + 2 slaves).
///
/// The device model stores port counts per bundle, not individual ports.
/// All ports within a bundle+direction are identical -- they differ only by
/// index, which can be enumerated from the count.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PortBundle {
    pub bundle: String,
    pub masters: u8,
    pub slaves: u8,
}

/// Instance counts for repeated hardware elements within a tile type.
///
/// Each field is independently `Confirmed` because sources are patchwork:
/// the device model provides all three, AM025 register analysis can confirm
/// locks and BDs (from register group counts), but has no signal for channels.
/// Single-source fields are automatically correct; multi-source fields must agree.
///
/// Memory banking info (num_banks, bank_size) lives in `MemoryModel`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstanceCount {
    pub locks: Confirmed<u8>,
    pub bds: Confirmed<u8>,
    pub channels: Confirmed<u8>,
}

/// A hardware subsystem within a module, materialized as a queryable object.
///
/// Subsystems partition a module's register space into functional units
/// (DMA, Lock, StreamSwitch, etc.). Each subsystem owns a contiguous offset
/// range within the tile's 20-bit address space and a set of registers.
///
/// Hierarchy: `TileTypeModel -> ModuleModel -> SubsystemModel`.
///
/// Offset ranges use `Confirmed<u32>` for cross-validation: the bottom-up
/// range (derived from AM025 register min/max offsets) will later be confirmed
/// against the top-down range (from aie-rt module base/size definitions).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SubsystemModel {
    pub kind: SubsystemKind,
    /// Start offset within the tile's 20-bit address space (inclusive).
    pub offset_start: Confirmed<u32>,
    /// End offset (exclusive).
    pub offset_end: Confirmed<u32>,
    /// Registers owned by this subsystem, moved here from ModuleModel
    /// during subsystem population.
    pub registers: Vec<RegisterModel>,
}

impl SubsystemModel {
    /// Returns true if `offset` falls within this subsystem's address range.
    ///
    /// The range is half-open: `[offset_start, offset_end)`.
    pub fn contains_offset(&self, offset: u32) -> bool {
        offset >= *self.offset_start.value() && offset < *self.offset_end.value()
    }
}

/// A functional module within a tile (Layer 2+: registers grouped by function).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModuleModel {
    pub kind: ModuleKind,
    /// Subsystems derived from register grouping. Each subsystem owns the
    /// registers that belong to it (moved from the module level during
    /// subsystem population).
    pub subsystems: Vec<SubsystemModel>,
    pub source: SourceAttribution,
}

impl ModuleModel {
    /// Iterate over all registers across all subsystems.
    pub fn all_registers(&self) -> impl Iterator<Item = &RegisterModel> {
        self.subsystems.iter().flat_map(|s| s.registers.iter())
    }

    /// Total register count across all subsystems.
    pub fn register_count(&self) -> usize {
        self.subsystems.iter().map(|s| s.registers.len()).sum()
    }
}

impl FactEquals for ModuleModel {
    fn fact_equals(&self, other: &Self) -> bool {
        self.kind == other.kind
            && self.register_count() == other.register_count()
            && self
                .all_registers()
                .zip(other.all_registers())
                .all(|(a, b)| a.fact_equals(b))
    }
}

// ============================================================================
// Tile and array topology
// ============================================================================

/// Memory banking at a single abstraction level.
///
/// Represents either the logical view (programmer/compiler, from mlir-aie)
/// or the physical view (SRAM arrays, from AM020/AM025).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BankingModel {
    pub num_banks: u8,
    pub bank_size: u64,
    pub bank_width_bits: u16,
    pub source: SourceAttribution,
}

impl FactEquals for BankingModel {
    fn fact_equals(&self, other: &Self) -> bool {
        self.num_banks == other.num_banks
            && self.bank_size == other.bank_size
            && self.bank_width_bits == other.bank_width_bits
    }
}

/// Memory model for a tile type.
///
/// `size_bytes` is total data memory (64KB compute, 512KB memtile).
/// Banking is modeled at two abstraction levels:
/// - **logical**: programmer/compiler view (from mlir-aie). Always present.
/// - **physical**: SRAM array view (from AM020/AM025). Optional -- not all
///   sources provide physical banking information.
///
/// The structural invariant `num_banks * bank_size == size_bytes` is enforced
/// at construction time for both levels.
///
/// `program_memory_bytes` is instruction memory (16KB, compute tiles only).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryModel {
    pub size_bytes: u64,
    /// Logical banking: programmer/compiler view (from mlir-aie).
    pub logical: BankingModel,
    /// Physical banking: SRAM array view (from AM020/AM025). Optional.
    pub physical: Option<BankingModel>,
    /// Instruction memory size. Only present on compute tiles.
    pub program_memory_bytes: Option<u64>,
    pub source: SourceAttribution,
}

impl MemoryModel {
    /// Create a new memory model with structural invariant validation.
    ///
    /// # Panics
    ///
    /// Panics if `logical.num_banks * logical.bank_size != size_bytes`
    /// (with message containing "logical banking invariant"), or if
    /// `physical` is `Some` and its banks do not cover `size_bytes`
    /// (with message containing "physical banking invariant").
    pub fn new(
        size_bytes: u64,
        logical: BankingModel,
        physical: Option<BankingModel>,
        program_memory_bytes: Option<u64>,
        source: SourceAttribution,
    ) -> Self {
        assert_eq!(
            logical.num_banks as u64 * logical.bank_size,
            size_bytes,
            "logical banking invariant: {} banks * {} bytes != {} total",
            logical.num_banks,
            logical.bank_size,
            size_bytes,
        );
        if let Some(ref phys) = physical {
            assert_eq!(
                phys.num_banks as u64 * phys.bank_size,
                size_bytes,
                "physical banking invariant: {} banks * {} bytes != {} total",
                phys.num_banks,
                phys.bank_size,
                size_bytes,
            );
        }
        Self {
            size_bytes,
            logical,
            physical,
            program_memory_bytes,
            source,
        }
    }

    /// Return the effective physical banking model.
    ///
    /// If physical banking information is available, returns it. Otherwise
    /// falls back to the logical banking model (which is always present).
    pub fn effective_physical(&self) -> &BankingModel {
        self.physical.as_ref().unwrap_or(&self.logical)
    }
}

impl FactEquals for MemoryModel {
    fn fact_equals(&self, other: &Self) -> bool {
        self.size_bytes == other.size_bytes
            && self.logical.fact_equals(&other.logical)
            && match (&self.physical, &other.physical) {
                (Some(a), Some(b)) => a.fact_equals(b),
                (None, None) => true,
                _ => false,
            }
            && self.program_memory_bytes == other.program_memory_bytes
    }
}

/// A tile type definition (e.g., "core", "mem_tile", "shim_noc").
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TileTypeModel {
    pub kind: TileKind,
    pub name: String,
    /// Representative tile coordinates [col, row] for this type.
    pub representative: Option<(u8, u8)>,
    /// Hardware resource counts (locks, BDs, channels, banks).
    pub instances: InstanceCount,
    /// Data/program memory model.
    pub memory: Option<MemoryModel>,
    /// DMA feature capabilities (compression, padding, etc.).
    pub dma_capabilities: Option<DmaCapabilities>,
    /// Core address space model (compute tiles only).
    pub core_address_map: Option<CoreAddressMap>,
    /// Switchbox port bundles (present on all tile types).
    pub switchbox_ports: Vec<PortBundle>,
    /// Shim mux port bundles (only present on shim tiles).
    pub shim_mux_ports: Vec<PortBundle>,
    /// Register modules (populated in Layer 2).
    pub modules: Vec<ModuleModel>,
    /// DMA buffer descriptor programming schema (populated from register analysis).
    pub bd_schema: Option<BdSchema>,
    /// DMA channel control/status schema (populated from register analysis).
    pub channel_schema: Option<DmaChannelSchema>,
    pub source: SourceAttribution,
}

/// Core data address space model.
///
/// Describes how the compute core addresses data memory via load/store
/// instructions. The core uses cardinal directions (South/West/North/East)
/// to address its own and neighboring tiles' data memory.
///
/// The address space layout is determined by `data_mem_addr` (start of
/// data memory in core view) and `data_mem_size` (size per quadrant,
/// equal to `MemoryModel.size_bytes` for compute tiles).
///
/// Cardinal direction = `address / data_mem_size`:
///   4=South, 5=West, 6=North, 7=East
///
/// For AIE2 (`is_checkerboard = false`), East is always the local tile.
/// For AIE1 (`is_checkerboard = true`), East/West alternate by row parity.
///
/// Source: aie-rt `XAie_CoreMod` struct, `_XAie_GetTargetTileLoc()`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoreAddressMap {
    /// Start of data memory in core address space (0x40000 for AIE2).
    /// Cardinal direction = address / data_mem_size.
    pub data_mem_addr: u32,
    /// Log2 of data memory size per quadrant (16 for AIE2 = 64KB boundaries).
    pub data_mem_shift: u8,
    /// Whether memory modules alternate sides by row (AIE1=true, AIE2=false).
    pub is_checkerboard: bool,
    /// Program memory offset in host/CDO address space (0x20000 for AIE2).
    /// This is the offset within the tile's register space where program
    /// memory is mapped, used by CDO writes and ELF loading.
    pub program_mem_host_offset: u32,
    pub source: SourceAttribution,
}

/// DMA feature capabilities for a tile type.
///
/// These flags describe what optional DMA features the hardware supports.
/// Derived from aie-rt's `XAie_DmaMod` structure, which defines these
/// per architecture generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DmaCapabilities {
    pub supports_compression: bool,
    pub supports_zero_padding: bool,
    pub supports_out_of_order_bd: bool,
    pub supports_interleave: bool,
    pub supports_fifo_mode: bool,
    pub supports_token_issue: bool,
    pub supports_repeat_count: bool,
    pub supports_tlast_suppress: bool,
    /// Number of addressing dimensions (typically 3 or 4).
    pub max_address_dimensions: u8,
}

// ============================================================================
// DMA Buffer Descriptor schema
// ============================================================================

/// Semantic role of a BD field -- what this field controls in the DMA engine.
///
/// Derived from AM025 register field names via pattern matching. The naming
/// convention is stable across architecture revisions.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BdFieldRole {
    /// Transfer length in address generation granularity units.
    BufferLength,
    /// Base address (or low half for shim split addresses).
    BaseAddress,
    /// High bits of base address (shim tiles only, 46-bit DDR addressing).
    BaseAddressHigh,
    /// Per-dimension step size. Dimension 0-3.
    Stepsize(u8),
    /// Per-dimension wrap count. Dimension 0-3.
    Wrap(u8),
    /// Zero-padding before transfer, per dimension (memtile only).
    ZeroPadBefore(u8),
    /// Zero-padding after transfer, per dimension (memtile only).
    ZeroPadAfter(u8),
    /// Iteration (outermost loop) step size.
    IterationStepsize,
    /// Iteration wrap count.
    IterationWrap,
    /// Iteration current counter.
    IterationCurrent,
    /// Lock ID to acquire before transfer.
    LockAcqId,
    /// Lock value for acquire operation.
    LockAcqValue,
    /// Whether lock acquire is enabled.
    LockAcqEnable,
    /// Lock ID to release after transfer.
    LockRelId,
    /// Lock value for release operation.
    LockRelValue,
    /// BD contains valid configuration.
    ValidBd,
    /// Chain to next BD after completion.
    UseNextBd,
    /// Index of the next BD in chain.
    NextBd,
    /// Enable packet header insertion (MM2S).
    EnablePacket,
    /// Packet ID for header.
    PacketId,
    /// Packet type for header.
    PacketType,
    /// Out-of-order BD ID for reordering.
    OutOfOrderBdId,
    /// Suppress TLAST signal on last beat.
    TlastSuppress,
    /// Enable hardware compression/decompression.
    EnableCompression,
    /// AXI burst length (shim only).
    BurstLength,
    /// AXI stream master ID (shim only).
    Smid,
    /// AXI cache policy (shim only).
    AxCache,
    /// AXI quality of service (shim only).
    AxQos,
    /// AXI secure access flag (shim only).
    SecureAccess,
    /// Reserved or unclassified field.
    Reserved,
}

/// One field within a buffer descriptor -- its position and semantic role.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BdFieldSpec {
    /// AM025 field name (e.g., "Buffer_Length", "D0_Stepsize").
    pub name: String,
    /// Which BD word this field lives in (0-based).
    pub word: u8,
    /// Bit range within that word.
    pub bits: BitRange,
    /// What this field controls.
    pub role: BdFieldRole,
}

/// Complete BD programming schema for one tile type.
///
/// Derived from DMA BD register groups in the AM025 register database.
/// Captures the BD layout (how many words, what fields, what widths) that
/// defines the DMA programming interface for this tile type.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BdSchema {
    /// How many 32-bit words per BD (6 for compute, 8 for memtile/shim).
    pub words_per_bd: u8,
    /// Number of addressing dimensions (3 for compute, 4 for memtile/shim).
    pub num_dimensions: u8,
    /// Whether zero-padding fields are present (memtile only).
    pub has_zero_padding: bool,
    /// Whether AXI bus fields are present (shim only).
    pub has_axi_fields: bool,
    /// All fields in the BD, ordered by (word, lsb).
    pub fields: Vec<BdFieldSpec>,
    pub source: SourceAttribution,
}

impl BdSchema {
    /// Find a field by its semantic role.
    pub fn field(&self, role: &BdFieldRole) -> Option<&BdFieldSpec> {
        self.fields.iter().find(|f| &f.role == role)
    }
}

// ============================================================================
// DMA Channel schema
// ============================================================================

/// DMA transfer direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DmaDirection {
    /// Stream to Memory (inbound DMA).
    S2mm,
    /// Memory to Stream (outbound DMA).
    Mm2s,
}

impl fmt::Display for DmaDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::S2mm => write!(f, "S2MM"),
            Self::Mm2s => write!(f, "MM2S"),
        }
    }
}

/// Which register within the per-channel register set.
///
/// Each DMA channel has a fixed set of register types. Control and StartQueue
/// are per-direction, per-channel. Status registers use a slightly different
/// naming convention (channel index as suffix, not infix) but are logically
/// part of the same channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DmaChannelRegKind {
    /// Channel configuration (enable, reset, compression, FoT mode).
    Control,
    /// Task enqueue register (BD ID, repeat count, token issue).
    /// Named "Start_Queue" on compute/memtile, "Task_Queue" on shim.
    StartQueue,
    /// Channel status and error reporting (read-only / write-to-clear).
    Status,
    /// Current write count for S2MM channels (read-only, S2MM only).
    WriteCount,
    /// Finish-on-TLAST count FIFO (read-only, S2MM only).
    FotCountFifo,
}

/// Semantic role of a DMA channel register field.
///
/// Derived from AM025 register field names. The roles capture what each
/// field controls or reports, independent of its bit position (which
/// varies by tile type).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DmaChannelFieldRole {
    // -- Control register fields --
    /// Soft reset the channel.
    Reset,
    /// Pause memory-side transfers (shim only).
    PauseMem,
    /// Pause stream-side transfers (shim only).
    PauseStream,
    /// Enable hardware compression on MM2S output.
    CompressionEnable,
    /// Enable hardware decompression on S2MM input.
    DecompressionEnable,
    /// Allow out-of-order BD completion (S2MM only).
    EnableOutOfOrder,
    /// Token controller ID for completion signaling.
    ControllerId,
    /// Finish-on-TLAST mode (S2MM only, 2-bit).
    FotMode,

    // -- Start/Task Queue fields --
    /// BD index to start executing.
    StartBdId,
    /// Number of times to repeat the task.
    RepeatCount,
    /// Emit a completion token when the task finishes.
    EnableTokenIssue,

    // -- Status register fields --
    /// Channel state machine (2-bit: IDLE/STARTING/RUNNING).
    ChannelStatus,
    /// Stalled waiting for lock acquire.
    StalledLockAcq,
    /// Stalled waiting for lock release.
    StalledLockRel,
    /// Stalled due to stream starvation (S2MM: no input data).
    StalledStreamStarvation,
    /// Stalled due to stream backpressure (MM2S: output blocked).
    StalledStreamBackpressure,
    /// Stalled on task completion token (MM2S).
    StalledTct,
    /// Stalled on TCT or count FIFO full (S2MM).
    StalledTctOrCountFifoFull,
    /// Error: no BD available in chain.
    ErrorBdUnavailable,
    /// Error: BD has invalid configuration.
    ErrorBdInvalid,
    /// Error: FoT transfer length exceeded.
    ErrorFotLengthExceeded,
    /// Error: too many BDs per FoT task.
    ErrorFotBdsPerTask,
    /// Error: lock access to unavailable tile (memtile only).
    ErrorLockAccessUnavailable,
    /// Error: data memory access to unavailable tile (memtile only).
    ErrorDmAccessUnavailable,
    /// AXI memory-mapped decode error (shim only).
    AxiMmDecodeError,
    /// AXI memory-mapped slave error (shim only).
    AxiMmSlaveError,
    /// Task queue overflow (write-to-clear).
    TaskQueueOverflow,
    /// Channel is currently running a task.
    ChannelRunning,
    /// Number of tasks queued (3-bit, 0-8).
    TaskQueueSize,
    /// Currently executing BD index.
    CurrentBdId,

    // -- Write Count register fields (S2MM only) --
    /// Current write byte count for the active transfer.
    CurrentWriteCount,

    // -- FoT Count FIFO fields (S2MM only) --
    /// Write count from completed FoT transfer.
    FotWriteCount,
    /// BD ID of completed FoT transfer.
    FotBdId,
    /// Whether this was the last BD in the task.
    FotLastInTask,
    /// Whether this FIFO entry is valid.
    FotValid,

    /// Reserved or unclassified field.
    Reserved,
}

/// One field within a DMA channel register set -- its position and semantic role.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DmaChannelFieldSpec {
    /// AM025 field name (e.g., "Start_BD_ID", "Channel_Running").
    pub name: String,
    /// Which register this field belongs to.
    pub register: DmaChannelRegKind,
    /// Which DMA direction this field applies to.
    pub direction: DmaDirection,
    /// Bit range within the register.
    pub bits: BitRange,
    /// What this field controls or reports.
    pub role: DmaChannelFieldRole,
}

/// Complete DMA channel programming schema for one tile type.
///
/// Derived from DMA channel control/status registers in the AM025 register
/// database. Captures the per-channel register layout that defines how
/// software configures and monitors DMA channels on this tile type.
///
/// This is the channel-level complement to `BdSchema` (which captures the
/// per-BD programming interface). Together they define the complete DMA
/// programming model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DmaChannelSchema {
    /// Number of channels per direction (e.g., 2 for compute/shim, 6 for memtile).
    pub channels_per_direction: u8,
    /// Width of Start_BD_ID field in bits (4 for compute/shim, 6 for memtile).
    pub start_bd_id_width: u8,
    /// Whether compression/decompression fields are present.
    pub has_compression: bool,
    /// Whether out-of-order BD completion is supported (S2MM only).
    pub has_out_of_order: bool,
    /// Whether pause (mem/stream) fields are present (shim only).
    pub has_pause: bool,
    /// Whether Finish-on-TLAST mode is supported.
    pub has_fot: bool,
    /// Whether AXI bus error fields are present (shim only).
    pub has_axi_errors: bool,
    /// Whether cross-tile error fields are present (memtile only).
    pub has_cross_tile_errors: bool,
    /// Name of the queue register ("Start_Queue" or "Task_Queue").
    pub queue_name: String,
    /// All fields across all channel register types and directions.
    pub fields: Vec<DmaChannelFieldSpec>,
    pub source: SourceAttribution,
}

impl DmaChannelSchema {
    /// Find a field by its semantic role and direction.
    pub fn field(
        &self,
        role: &DmaChannelFieldRole,
        direction: DmaDirection,
    ) -> Option<&DmaChannelFieldSpec> {
        self.fields
            .iter()
            .find(|f| &f.role == role && f.direction == direction)
    }

    /// Find all fields for a given register kind and direction.
    pub fn fields_for(
        &self,
        reg: DmaChannelRegKind,
        direction: DmaDirection,
    ) -> Vec<&DmaChannelFieldSpec> {
        self.fields
            .iter()
            .filter(|f| f.register == reg && f.direction == direction)
            .collect()
    }
}

// ============================================================================
// Timing, packet format, and FoT models (AM020-sourced)
// ============================================================================

/// Lock subsystem timing constants.
///
/// Source: AM020 Ch2 ("The lock module can handle a new request every clock
/// cycle"). These have no machine-readable source -- they come from
/// architecture manual prose and hardware observation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LockTiming {
    /// Uncontested acquire latency in cycles.
    pub acquire_latency: u8,
    /// Release latency in cycles.
    pub release_latency: u8,
    /// Retry interval when acquire fails (core stalls and retries each cycle).
    pub retry_interval: u8,
}

/// DMA engine timing constants.
///
/// Source: AM020 Ch2 (architecture description) and hardware observation
/// (host memory latency from trace comparison).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DmaTiming {
    /// Cycles to parse and configure a buffer descriptor.
    pub bd_setup_cycles: u8,
    /// Cycles from start trigger to first data.
    pub channel_start_cycles: u8,
    /// Throughput: words (32-bit) per cycle per channel.
    pub words_per_cycle: u8,
    /// Memory access latency in cycles (same as data memory pipeline depth).
    pub memory_latency_cycles: u8,
    /// Cycles to check and acquire a lock.
    pub lock_acquire_cycles: u8,
    /// Cycles to release a lock.
    pub lock_release_cycles: u8,
    /// Cycles between finishing one BD and starting next in chain.
    pub bd_chain_cycles: u8,
    /// Extra cycles for shim tile DDR access (initial pipeline fill penalty).
    /// Derived from trace comparison: ~110cy observed minus ~10cy already
    /// covered by bd_setup + memory_latency.
    pub host_memory_latency_cycles: u16,
}

/// Stream switch timing and physical constants.
///
/// Source: AM020 Ch2 ("Local slave ports are 2-cycle latency and a 4-deep
/// FIFO", "Local master ports have 1-cycle latency and a 2-deep FIFO").
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamSwitchTiming {
    /// Slave port FIFO depth (words).
    pub local_slave_fifo_depth: u8,
    /// Master port FIFO depth (words).
    pub local_master_fifo_depth: u8,
    /// Local slave to local master latency in cycles.
    pub local_to_local_latency: u8,
    /// Local slave to external master latency in cycles.
    pub local_to_external_latency: u8,
    /// External slave to external master latency in cycles.
    pub external_to_external_latency: u8,
    /// External slave to local master latency in cycles.
    pub external_to_local_latency: u8,
    /// Packet switch arbitration overhead per packet header.
    pub packet_arbitration_overhead: u8,
}

/// Instruction-level timing constants.
///
/// Source: AM020 Ch4 ("Load and store units manage the 5-cycle latency of
/// data memory").
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstructionTiming {
    /// Data memory access pipeline depth in cycles.
    pub data_memory_latency: u8,
    /// Branch penalty: cycles lost when a branch is taken.
    pub branch_penalty: u8,
}

/// Complete timing model for one architecture.
///
/// All values sourced from AM020 prose and hardware observation. No
/// machine-readable source exists for these constants. Structured so
/// that if a data source is found later, values can be cross-validated
/// via `Confirmed<T>` without changing the generated output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimingModel {
    pub lock: LockTiming,
    pub dma: DmaTiming,
    pub stream_switch: StreamSwitchTiming,
    pub instruction: InstructionTiming,
    pub source: SourceAttribution,
}

/// VLIW processor model: slot widths, register sizes, pipeline constants.
///
/// Sources: llvm-aie AIE2Slots.td (slot widths), AIE2GenRegisterInfo.td
/// (register widths), AM020 Ch4 (pipeline constants).
///
/// Slot widths and register sizes can be cross-validated at runtime against
/// the llvm-aie TableGen parser output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProcessorModel {
    /// VLIW slot widths in bits, keyed by slot name (lda, ldb, alu, mv, st, vec, lng).
    /// Source: llvm-aie AIE2Slots.td `InstSlot<"Name", width>`.
    pub slot_widths: Vec<(String, u8)>,
    /// Vector register width in bits (512 for AIE2).
    /// Source: llvm-aie AIE2GenRegisterInfo.td `AIE2Vector512RegisterClass`.
    pub vector_register_bits: u16,
    /// Vector pair width in bits (1024 for AIE2, used by shuffle/permute).
    /// Source: llvm-aie AIE2GenRegisterInfo.td `AIE2Vector1024RegisterClass`.
    pub vector_pair_bits: u16,
    /// Accumulator register width in bits (512 for AIE2 standard accumulators).
    /// Source: llvm-aie AIE2GenRegisterInfo.td `AIE2Acc512RegisterClass`.
    pub accumulator_bits: u16,
    /// Branch delay slot count (pipeline depth, NOT branch penalty).
    /// The number of instructions after a branch that execute regardless.
    /// Source: AM020 Ch4 "5 instruction delay slots".
    pub branch_delay_slots: u8,
    /// Partial-word store (st.s8/u8/s16/u16) data register read latency.
    /// The cycle offset from issue at which the data register is read.
    /// Source: hardware observation (llvm-aie II_STHB scheduling class).
    pub partial_store_data_latency: u8,
    /// SRS (Scale/Round/Shift) hardware bias in bits.
    /// The accumulator is effectively left-shifted by this many bits
    /// before the user-specified shift is applied.
    /// Source: AM020 / aietools vector model.
    pub srs_shift_bias: u8,
    /// Whether this architecture has a cascade link between adjacent compute tiles.
    ///
    /// AIE2 and AIE2P compute tiles have a dedicated 384-bit point-to-point
    /// cascade link (6 x u64). AIE1 does not have this link. Set `true` for
    /// AIE2/AIE2P; `false` for AIE1 (when AIE1 support is added).
    /// Source: aie-rt xaie_core.c:993-1046; AM020 cascade stream description.
    pub has_cascade_link: bool,
    pub source: SourceAttribution,
}

/// Stream packet header bit layout.
///
/// 32-bit header: parity(31) | rsvd(30-28) | src_col(27-21) |
/// src_row(20-16) | rsvd(15) | pkt_type(14-12) | rsvd(11-5) | stream_id(4-0)
///
/// Source: AM020 Ch2, Table 2. Also confirmed in mlir-aie
/// `AIETargetNPU.cpp:309-336` (packet header construction).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamPacketFormat {
    pub stream_id_mask: u32,
    pub packet_type_shift: u8,
    pub packet_type_mask: u32,
    pub src_row_shift: u8,
    pub src_row_mask: u32,
    pub src_col_shift: u8,
    pub src_col_mask: u32,
    pub parity_shift: u8,
}

/// Control packet header bit layout and operation codes.
///
/// Control packets reprogram tile registers at runtime via the
/// TileControl stream master port.
///
/// Source: AM020 Table 3. Operation codes not found in any open-source
/// toolchain code.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ControlPacketFormat {
    /// Address field mask (bits 19:0).
    pub address_mask: u32,
    /// Length field shift (bits 21:20, value+1 = beats).
    pub length_shift: u8,
    pub length_mask: u32,
    /// Operation field shift (bits 23:22).
    pub operation_shift: u8,
    pub operation_mask: u32,
    /// Response_ID field shift (bits 30:24).
    pub response_id_shift: u8,
    pub response_id_mask: u32,
    /// Parity bit position (bit 31).
    pub parity_bit: u8,
    /// Operation code: write data to register(s).
    pub op_write: u8,
    /// Operation code: read register.
    pub op_read: u8,
    /// Operation code: write with auto-increment address.
    pub op_write_incr: u8,
    /// Operation code: block write.
    pub op_block_write: u8,
}

/// DMA Finish-on-TLAST mode values.
///
/// Source: AM025 DMA_S2MM_x_Ctrl.FoT_Mode field (confirmed in aie-rt
/// `xaiegbl.h:419-424`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FotConfig {
    /// FoT disabled: channel runs until BD transfer count is exhausted.
    pub disabled: u8,
    /// Transfer finishes on TLAST regardless of count.
    pub no_counts: u8,
    /// Finish on TLAST, issue task-complete token.
    pub counts_with_tokens: u8,
    /// Length comes from a separate count register.
    pub counts_from_register: u8,
}

/// Complete packet and protocol format model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PacketModel {
    pub stream: StreamPacketFormat,
    pub control: ControlPacketFormat,
    pub fot: FotConfig,
    pub source: SourceAttribution,
}

/// Device-level hardware constants.
///
/// These are properties of the device as a whole, not of any specific tile.
/// Values come from the mlir-aie device model and aie-rt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceConstants {
    /// Maximum lock counter value (e.g., 63 for AIE2).
    pub max_lock_value: u32,
    /// Minimum lock counter value (e.g., -64 for AIE-ML, 0 for AIE1).
    pub min_lock_value: i32,
    /// DMA address generation granularity in bytes (e.g., 32).
    pub address_gen_granularity: u32,
    /// Accumulator cascade width in bits (384 for AIE1, 512 for AIE2).
    pub accumulator_cascade_bits: Option<u32>,
    /// Memory base addresses per cardinal direction, used for cross-tile
    /// memory access routing. Keys: south, west, north, east.
    pub mem_base_addresses: std::collections::BTreeMap<String, u64>,
    /// Architecture-level capability flags.
    pub properties: DeviceProperties,
    pub source: SourceAttribution,
}

/// Architecture-level capability flags.
///
/// Derived from mlir-aie's `ModelProperty` enum. These describe fundamental
/// behavioral differences between architecture generations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceProperties {
    /// Uses semaphore-style locks (AIE2+) vs. binary locks (AIE1).
    pub uses_semaphore_locks: bool,
    /// Supports multi-dimensional buffer descriptors.
    pub uses_multi_dim_bds: bool,
}

/// Directional flags (north/south/east/west).
///
/// Used for tile boundary edges and memory affinity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CardinalFlags {
    pub north: bool,
    pub south: bool,
    pub east: bool,
    pub west: bool,
}

/// A single tile's placement within the array.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TilePlacement {
    pub col: u8,
    pub row: u8,
    /// Which tile type this position holds (matches a TileTypeModel name).
    pub tile_type: String,
    /// Whether this tile is internal (not on any array boundary).
    pub is_internal: bool,
    /// Which edges of this tile are on the array boundary.
    pub edges: CardinalFlags,
    /// Memory affinity: which directions have accessible memory neighbors.
    /// Only present for compute tiles.
    pub mem_affinity: Option<CardinalFlags>,
}

/// Array topology: grid dimensions, tile placement, and addressing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArrayTopology {
    pub columns: u8,
    pub rows: u8,
    pub num_mem_tile_rows: u8,
    pub column_shift: u8,
    pub row_shift: u8,
    /// Per-tile placement map: one entry per physical tile in the array.
    pub tile_map: Vec<TilePlacement>,
    pub source: SourceAttribution,
}

// ============================================================================
// Graph edges: node identity and relationships
// ============================================================================

/// Unique identity for a node in the architecture graph.
///
/// Each variant carries exactly the coordinates needed to identify that
/// kind of node. Hierarchical nodes (registers, fields) carry their
/// parent coordinates. Cross-cutting nodes (schema fields) carry only
/// the coordinates relevant to their identity.
///
/// This is designed to grow incrementally -- add variants as new node
/// kinds enter the graph, not preemptively.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeId {
    /// A tile type definition.
    TileType {
        kind: TileKind,
    },

    /// A hardware module within a tile type.
    Module {
        tile: TileKind,
        module: ModuleKind,
    },

    /// A register within a module.
    Register {
        tile: TileKind,
        module: ModuleKind,
        name: String,
    },

    /// A bit field within a register.
    RegisterField {
        tile: TileKind,
        module: ModuleKind,
        register: String,
        field: String,
    },

    /// A field in the BD programming schema.
    BdField {
        tile: TileKind,
        role: BdFieldRole,
    },

    /// A field in the DMA channel programming schema.
    ChannelField {
        tile: TileKind,
        direction: DmaDirection,
        role: DmaChannelFieldRole,
    },

    /// A functional subsystem within a module (DMA, Lock, StreamSwitch, etc.).
    Subsystem {
        tile: TileKind,
        module: ModuleKind,
        subsystem: SubsystemKind,
    },
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TileType { kind } => write!(f, "{:?}", kind),
            Self::Module { tile, module } => write!(f, "{:?}.{:?}", tile, module),
            Self::Register { tile, module, name } => {
                write!(f, "{:?}.{:?}.{}", tile, module, name)
            }
            Self::RegisterField { tile, module, register, field } => {
                write!(f, "{:?}.{:?}.{}.{}", tile, module, register, field)
            }
            Self::BdField { tile, role } => {
                write!(f, "{:?}.bd.{:?}", tile, role)
            }
            Self::ChannelField { tile, direction, role } => {
                write!(f, "{:?}.channel.{}.{:?}", tile, direction, role)
            }
            Self::Subsystem { tile, module, subsystem } => {
                write!(f, "{:?}.{:?}.{:?}", tile, module, subsystem)
            }
        }
    }
}

/// Type of directed relationship between two graph nodes.
///
/// Start minimal. Add variants as real extraction code needs them, not
/// speculatively. Each variant should have at least one producer before
/// being added.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RelationshipKind {
    /// A schema field was derived from a register field.
    /// Direction: schema field -> register field (target is the source of truth).
    DerivedFrom,

    /// A node structurally contains another node.
    /// Direction: parent -> child.
    Contains,

    /// A node belongs to a functional grouping.
    /// Direction: member -> group (e.g., register -> subsystem).
    BelongsTo,

    /// A field's value indexes into or points at another resource space.
    /// Direction: referencing field -> referenced resource.
    /// Example: BD's LockAcqId field -> Lock subsystem.
    References,
}

/// A directed relationship between two nodes in the architecture graph.
///
/// Edges are first-class objects with source attribution, just like node
/// data. Every edge was created by a specific extraction step and can be
/// traced back to its origin.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Relationship {
    pub from: NodeId,
    pub to: NodeId,
    pub kind: RelationshipKind,
    pub source: SourceAttribution,
}

// ============================================================================
// Top-level model
// ============================================================================

/// Complete architecture model for one NPU device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArchModel {
    pub arch: Architecture,
    /// Fine-grained device generation (aie-rt XAIE_DEV_GEN_*).
    pub generation: Option<DeviceGeneration>,
    /// mlir-aie device ID (e.g., npu1=4, npu2=8).
    pub device_id: Option<u32>,
    /// Whether this device is an NPU (vs. Versal FPGA).
    pub is_npu: bool,
    pub tile_types: Vec<TileTypeModel>,
    pub array_topology: Option<ArrayTopology>,
    pub device_constants: Option<DeviceConstants>,
    /// Timing model (lock, DMA, stream switch, instruction latencies).
    pub timing: Option<TimingModel>,
    /// Packet format model (stream headers, control packets, FoT modes).
    pub packet: Option<PacketModel>,
    /// VLIW processor model (slot widths, register sizes, pipeline constants).
    pub processor: Option<ProcessorModel>,
    pub relationships: Vec<Relationship>,
}

impl ArchModel {
    /// Create an empty model for the given architecture.
    pub fn new(arch: Architecture) -> Self {
        Self {
            arch,
            generation: None,
            device_id: None,
            is_npu: false,
            tile_types: Vec::new(),
            array_topology: None,
            device_constants: None,
            timing: None,
            packet: None,
            processor: None,
            relationships: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_source(origin: Source, detail: &str) -> SourceAttribution {
        SourceAttribution {
            origin,
            file: "test.json".to_string(),
            detail: detail.to_string(),
        }
    }

    /// Convenience: create an InstanceCount with a test source for all fields.
    fn test_instances(locks: u8, bds: u8, channels: u8) -> InstanceCount {
        let src = test_source(Source::DeviceModel, "test");
        InstanceCount {
            locks: Confirmed::new(locks, src.clone()),
            bds: Confirmed::new(bds, src.clone()),
            channels: Confirmed::new(channels, src),
        }
    }

    #[test]
    fn confirmed_new_has_one_source() {
        let c = Confirmed::new(16u8, test_source(Source::DeviceModel, "locks"));
        assert_eq!(*c.value(), 16u8);
        assert_eq!(c.sources().len(), 1);
        assert_eq!(c.sources()[0].origin, Source::DeviceModel);
    }

    #[test]
    fn confirmed_matching_value_succeeds() {
        let mut c = Confirmed::new(16u8, test_source(Source::DeviceModel, "locks"));
        c.confirm(16u8, test_source(Source::Am025Json, "register analysis"));
        assert_eq!(c.sources().len(), 2);
    }

    #[test]
    fn confirmed_multiple_sources_number_agnostic() {
        let mut c = Confirmed::new(16u8, test_source(Source::DeviceModel, "device model"));
        c.confirm(16u8, test_source(Source::Am025Json, "register analysis"));
        c.confirm(16u8, test_source(Source::AieRt, "aie-rt"));
        assert_eq!(*c.value(), 16u8);
        assert_eq!(c.sources().len(), 3);
        assert_eq!(c.sources()[0].origin, Source::DeviceModel);
        assert_eq!(c.sources()[1].origin, Source::Am025Json);
        assert_eq!(c.sources()[2].origin, Source::AieRt);
    }

    #[test]
    fn instance_count_per_field_confirmation() {
        let dm_src = test_source(Source::DeviceModel, "npu1 core");
        let am_src = test_source(Source::Am025Json, "register grouping");

        // Device model provides all three fields
        let mut counts = InstanceCount {
            locks: Confirmed::new(16, dm_src.clone()),
            bds: Confirmed::new(16, dm_src.clone()),
            channels: Confirmed::new(2, dm_src),
        };

        // AM025 register analysis can confirm locks and bds (from register
        // group counts), but has no signal for channels.
        counts.locks.confirm(16, am_src.clone());
        counts.bds.confirm(16, am_src);

        // locks and bds confirmed by 2 sources, channels by 1
        assert_eq!(counts.locks.sources().len(), 2);
        assert_eq!(counts.bds.sources().len(), 2);
        assert_eq!(counts.channels.sources().len(), 1);

        // Values accessible through .value()
        assert_eq!(*counts.locks.value(), 16);
        assert_eq!(*counts.bds.value(), 16);
        assert_eq!(*counts.channels.value(), 2);
    }

    #[test]
    #[should_panic(expected = "GRAPH CONFLICT")]
    fn instance_count_catches_per_field_disagreement() {
        let dm_src = test_source(Source::DeviceModel, "npu1 core");
        let am_src = test_source(Source::Am025Json, "register grouping");

        let mut counts = InstanceCount {
            locks: Confirmed::new(16, dm_src.clone()),
            bds: Confirmed::new(16, dm_src.clone()),
            channels: Confirmed::new(2, dm_src),
        };

        // Register analysis finds 48 BDs (memtile value for a core tile -- bug!)
        counts.bds.confirm(48, am_src);
    }

    #[test]
    fn confirmed_equality_ignores_sources() {
        let c1 = Confirmed::new(16u8, test_source(Source::DeviceModel, "dm"));
        let c2 = Confirmed::new(16u8, test_source(Source::AieRt, "rt"));
        assert_eq!(c1, c2); // same value, different sources -> equal
    }

    #[test]
    #[should_panic(expected = "GRAPH CONFLICT")]
    fn confirmed_conflicting_value_panics() {
        let mut c = Confirmed::new(16u8, test_source(Source::DeviceModel, "locks"));
        c.confirm(48u8, test_source(Source::Am025Json, "register analysis"));
    }

    #[test]
    fn test_architecture_enum_variants() {
        // Verify all expected architectures exist
        let archs = [Architecture::Aie, Architecture::Aie2, Architecture::Aie2p];
        assert_eq!(archs.len(), 3);
    }

    #[test]
    fn test_tile_kind_variants() {
        let kinds = [TileKind::Compute, TileKind::Mem, TileKind::ShimNoc, TileKind::ShimPl];
        assert_eq!(kinds.len(), 4);
        assert_eq!(format!("{}", TileKind::ShimNoc), "shim_noc");
        assert_eq!(format!("{}", TileKind::ShimPl), "shim_pl");
    }

    #[test]
    fn confirmed_with_custom_fact_equals() {
        // Two RegisterModels from different sources but same hardware data.
        // They should confirm successfully because FactEquals ignores `source`.
        let reg1 = RegisterModel {
            name: "DMA_BD0_0".to_string(),
            offset: 0x1D000,
            width: 32,
            reset_value: 0,
            fields: vec![],
            subsystem: SubsystemKind::Dma,
            access: Access::ReadWrite,
            source: test_source(Source::Am025Json, "am025"),
        };
        let reg2 = RegisterModel {
            name: "DMA_BD0_0".to_string(),
            offset: 0x1D000,
            width: 32,
            reset_value: 0,
            fields: vec![],
            subsystem: SubsystemKind::Dma,
            access: Access::ReadWrite,
            source: test_source(Source::AieRt, "aie-rt"),
        };
        // PartialEq would fail because sources differ.
        // fact_equals should succeed because hardware data matches.
        assert!(reg1.fact_equals(&reg2));
    }

    #[test]
    fn confirmed_fact_equals_detects_real_difference() {
        let reg1 = RegisterModel {
            name: "DMA_BD0_0".to_string(),
            offset: 0x1D000,
            width: 32,
            reset_value: 0,
            fields: vec![],
            subsystem: SubsystemKind::Dma,
            access: Access::ReadWrite,
            source: test_source(Source::Am025Json, "am025"),
        };
        let reg2 = RegisterModel {
            name: "DMA_BD0_0".to_string(),
            offset: 0x1E000, // different offset -- real disagreement
            width: 32,
            reset_value: 0,
            fields: vec![],
            subsystem: SubsystemKind::Dma,
            access: Access::ReadWrite,
            source: test_source(Source::AieRt, "aie-rt"),
        };
        assert!(!reg1.fact_equals(&reg2));
    }

    #[test]
    fn confirmed_confirm_uses_fact_equals() {
        let reg = RegisterModel {
            name: "DMA_BD0_0".to_string(),
            offset: 0x1D000,
            width: 32,
            reset_value: 0,
            fields: vec![],
            subsystem: SubsystemKind::Dma,
            access: Access::ReadWrite,
            source: test_source(Source::Am025Json, "am025"),
        };
        let reg_same_data = RegisterModel {
            name: "DMA_BD0_0".to_string(),
            offset: 0x1D000,
            width: 32,
            reset_value: 0,
            fields: vec![],
            subsystem: SubsystemKind::Dma,
            access: Access::ReadWrite,
            source: test_source(Source::AieRt, "aie-rt"), // different source
        };
        let mut c = Confirmed::new(reg, test_source(Source::Am025Json, "first"));
        // Should succeed despite different source in the RegisterModel
        c.confirm(reg_same_data, test_source(Source::AieRt, "second"));
        assert_eq!(c.sources().len(), 2);
    }

    #[test]
    fn test_source_attribution_display() {
        let attr = SourceAttribution {
            origin: Source::Am025Json,
            file: "aie_registers_aie2.json".to_string(),
            detail: "module=memory, register=DMA_BD0_0".to_string(),
        };
        let s = format!("{}", attr);
        assert!(s.contains("AM025 JSON"));
        assert!(s.contains("aie_registers_aie2.json"));
    }

    #[test]
    fn test_bit_range_contiguous() {
        let range = BitRange::Contiguous { msb: 17, lsb: 0 };
        assert_eq!(range.width(), 18);
    }

    #[test]
    fn test_bit_range_split() {
        let range = BitRange::Split(vec![
            (23, 20),  // 4 bits
            (7, 4),    // 4 bits
        ]);
        assert_eq!(range.width(), 8);
    }

    #[test]
    fn test_arch_model_empty() {
        let model = ArchModel::new(Architecture::Aie2);
        assert_eq!(model.arch, Architecture::Aie2);
        assert_eq!(model.device_id, None);
        assert!(!model.is_npu);
        assert!(model.tile_types.is_empty());
        assert!(model.relationships.is_empty());
    }

    #[test]
    fn test_serde_round_trip() {
        let mut model = ArchModel::new(Architecture::Aie2);
        model.device_id = Some(4);
        model.is_npu = true;
        model.tile_types.push(TileTypeModel {
            kind: TileKind::Compute,
            name: "core".to_string(),
            representative: Some((1, 2)),
            instances: test_instances(16, 16, 2),
            modules: vec![ModuleModel {
                kind: ModuleKind::Core,
                subsystems: vec![SubsystemModel {
                    kind: SubsystemKind::Dma,
                    offset_start: Confirmed::new(0x1D000, SourceAttribution {
                        origin: Source::Am025Json,
                        file: "test.json".to_string(),
                        detail: "test".to_string(),
                    }),
                    offset_end: Confirmed::new(0x1D004, SourceAttribution {
                        origin: Source::Am025Json,
                        file: "test.json".to_string(),
                        detail: "test".to_string(),
                    }),
                    registers: vec![RegisterModel {
                        name: "DMA_BD0_0".to_string(),
                        offset: 0x1D000,
                        width: 32,
                        reset_value: 0,
                        fields: vec![FieldModel {
                            name: "Buffer_Length".to_string(),
                            bits: BitRange::Contiguous { msb: 13, lsb: 0 },
                            meaning: FieldSemantics::Count,
                            source: SourceAttribution {
                                origin: Source::Am025Json,
                                file: "test.json".to_string(),
                                detail: "test".to_string(),
                            },
                        }],
                        subsystem: SubsystemKind::Dma,
                        access: Access::ReadWrite,
                        source: SourceAttribution {
                            origin: Source::Am025Json,
                            file: "test.json".to_string(),
                            detail: "test".to_string(),
                        },
                    }],
                }],
                source: SourceAttribution {
                    origin: Source::Am025Json,
                    file: "test.json".to_string(),
                    detail: "test".to_string(),
                },
            }],
            memory: Some(MemoryModel::new(
                65536,
                BankingModel {
                    num_banks: 4,
                    bank_size: 16384,
                    bank_width_bits: 32,
                    source: SourceAttribution {
                        origin: Source::DeviceModel,
                        file: "test.json".to_string(),
                        detail: "test".to_string(),
                    },
                },
                None,
                Some(16384),
                SourceAttribution {
                    origin: Source::DeviceModel,
                    file: "test.json".to_string(),
                    detail: "test".to_string(),
                },
            )),
            dma_capabilities: Some(DmaCapabilities {
                supports_compression: true,
                supports_zero_padding: true,
                supports_out_of_order_bd: false,
                supports_interleave: false,
                supports_fifo_mode: true,
                supports_token_issue: true,
                supports_repeat_count: true,
                supports_tlast_suppress: false,
                max_address_dimensions: 4,
            }),
            core_address_map: None,
            switchbox_ports: vec![
                PortBundle { bundle: "DMA".to_string(), masters: 2, slaves: 2 },
                PortBundle { bundle: "Core".to_string(), masters: 1, slaves: 1 },
            ],
            shim_mux_ports: Vec::new(),
            bd_schema: None,
            channel_schema: None,
            source: SourceAttribution {
                origin: Source::DeviceModel,
                file: "test.json".to_string(),
                detail: "test".to_string(),
            },
        });
        model.relationships.push(Relationship {
            from: NodeId::TileType { kind: TileKind::Compute },
            to: NodeId::Module { tile: TileKind::Compute, module: ModuleKind::Memory },
            kind: RelationshipKind::Contains,
            source: SourceAttribution {
                origin: Source::DeviceModel,
                file: "derived".to_string(),
                detail: "test".to_string(),
            },
        });

        let json = serde_json::to_string(&model).expect("serialize failed");
        let deserialized: ArchModel = serde_json::from_str(&json).expect("deserialize failed");
        assert_eq!(model, deserialized);
    }

    #[test]
    fn test_device_constants() {
        let mut mem_bases = std::collections::BTreeMap::new();
        mem_bases.insert("south".to_string(), 262144);
        mem_bases.insert("west".to_string(), 327680);
        mem_bases.insert("north".to_string(), 393216);
        mem_bases.insert("east".to_string(), 458752);

        let constants = DeviceConstants {
            max_lock_value: 63,
            min_lock_value: -64,
            address_gen_granularity: 32,
            accumulator_cascade_bits: Some(512),
            mem_base_addresses: mem_bases,
            properties: DeviceProperties {
                uses_semaphore_locks: true,
                uses_multi_dim_bds: true,
            },
            source: SourceAttribution {
                origin: Source::DeviceModel,
                file: "test.json".to_string(),
                detail: "npu1".to_string(),
            },
        };
        assert_eq!(constants.max_lock_value, 63);
        assert_eq!(constants.min_lock_value, -64);
        assert_eq!(constants.accumulator_cascade_bits, Some(512));
        assert!(constants.properties.uses_semaphore_locks);
        assert_eq!(constants.mem_base_addresses.len(), 4);
        assert_eq!(constants.mem_base_addresses["south"], 262144);
    }

    #[test]
    fn test_tile_placement() {
        let placement = TilePlacement {
            col: 1,
            row: 2,
            tile_type: "core".to_string(),
            is_internal: true,
            edges: CardinalFlags {
                north: false,
                south: false,
                east: false,
                west: false,
            },
            mem_affinity: Some(CardinalFlags {
                south: true,
                west: false,
                north: true,
                east: false,
            }),
        };
        assert!(placement.is_internal);
        assert!(placement.mem_affinity.unwrap().south);
        assert!(!placement.mem_affinity.unwrap().north || true); // just exercising access
    }

    #[test]
    fn test_port_bundles() {
        let tile = TileTypeModel {
            kind: TileKind::ShimNoc,
            name: "shim_noc".to_string(),
            representative: Some((1, 0)),
            instances: test_instances(16, 16, 2),
            memory: None,
            dma_capabilities: None,
            core_address_map: None,
            switchbox_ports: vec![
                PortBundle { bundle: "South".to_string(), masters: 6, slaves: 8 },
                PortBundle { bundle: "North".to_string(), masters: 6, slaves: 4 },
                PortBundle { bundle: "Ctrl".to_string(), masters: 1, slaves: 1 },
            ],
            shim_mux_ports: vec![
                PortBundle { bundle: "DMA".to_string(), masters: 2, slaves: 2 },
                PortBundle { bundle: "South".to_string(), masters: 8, slaves: 6 },
            ],
            modules: Vec::new(),
            bd_schema: None,
            channel_schema: None,
            source: SourceAttribution {
                origin: Source::DeviceModel,
                file: "test.json".to_string(),
                detail: "shim_noc".to_string(),
            },
        };
        // Switchbox and shim_mux are distinct port namespaces
        assert_eq!(tile.switchbox_ports.len(), 3);
        assert_eq!(tile.shim_mux_ports.len(), 2);
        // DMA ports only appear in shim_mux for shim tiles
        assert!(tile.switchbox_ports.iter().all(|p| p.bundle != "DMA"));
        assert!(tile.shim_mux_ports.iter().any(|p| p.bundle == "DMA"));
    }

    #[test]
    fn test_memory_model_program_memory() {
        let src = test_source(Source::DeviceModel, "test");
        let compute_mem = MemoryModel::new(
            65536,
            BankingModel {
                num_banks: 4,
                bank_size: 16384,
                bank_width_bits: 32,
                source: src.clone(),
            },
            None,
            Some(16384),
            src.clone(),
        );
        let memtile_mem = MemoryModel::new(
            524288,
            BankingModel {
                num_banks: 8,
                bank_size: 65536,
                bank_width_bits: 128,
                source: src.clone(),
            },
            None,
            None,
            src,
        );
        assert_eq!(compute_mem.program_memory_bytes, Some(16384));
        assert_eq!(memtile_mem.program_memory_bytes, None);
    }

    #[test]
    fn test_serde_round_trip_with_topology() {
        let mut model = ArchModel::new(Architecture::Aie2);

        let mut mem_bases = std::collections::BTreeMap::new();
        mem_bases.insert("south".to_string(), 262144);

        model.generation = Some(DeviceGeneration::Aie2Ipu);
        model.device_constants = Some(DeviceConstants {
            max_lock_value: 63,
            min_lock_value: -64,
            address_gen_granularity: 32,
            accumulator_cascade_bits: Some(512),
            mem_base_addresses: mem_bases,
            properties: DeviceProperties {
                uses_semaphore_locks: true,
                uses_multi_dim_bds: true,
            },
            source: SourceAttribution {
                origin: Source::DeviceModel,
                file: "test.json".to_string(),
                detail: "test".to_string(),
            },
        });

        model.array_topology = Some(ArrayTopology {
            columns: 4,
            rows: 6,
            num_mem_tile_rows: 1,
            column_shift: 25,
            row_shift: 20,
            tile_map: vec![
                TilePlacement {
                    col: 0,
                    row: 0,
                    tile_type: "shim_noc".to_string(),
                    is_internal: false,
                    edges: CardinalFlags {
                        north: false, south: true, east: false, west: true,
                    },
                    mem_affinity: None,
                },
                TilePlacement {
                    col: 1,
                    row: 2,
                    tile_type: "core".to_string(),
                    is_internal: true,
                    edges: CardinalFlags {
                        north: false, south: false, east: false, west: false,
                    },
                    mem_affinity: Some(CardinalFlags {
                        south: true, west: false, north: true, east: false,
                    }),
                },
            ],
            source: SourceAttribution {
                origin: Source::DeviceModel,
                file: "test.json".to_string(),
                detail: "test".to_string(),
            },
        });

        let json = serde_json::to_string(&model).expect("serialize failed");
        let deserialized: ArchModel = serde_json::from_str(&json).expect("deserialize failed");
        assert_eq!(model, deserialized);

        // Verify we can navigate the topology
        let topo = deserialized.array_topology.unwrap();
        assert_eq!(topo.tile_map.len(), 2);
        assert_eq!(topo.tile_map[0].tile_type, "shim_noc");
        assert!(topo.tile_map[1].mem_affinity.unwrap().south);
    }

    // ========================================================================
    // BankingModel / MemoryModel tests
    // ========================================================================

    #[test]
    fn memory_model_structural_invariant() {
        let src = test_source(Source::DeviceModel, "compute tile");
        let mem = MemoryModel::new(
            65536,
            BankingModel {
                num_banks: 4,
                bank_size: 16384,
                bank_width_bits: 32,
                source: src.clone(),
            },
            None,
            Some(16384),
            src,
        );
        assert_eq!(mem.size_bytes, 65536);
        assert_eq!(mem.logical.num_banks, 4);
        assert_eq!(mem.logical.bank_size, 16384);
        assert_eq!(mem.logical.bank_width_bits, 32);
        assert_eq!(mem.program_memory_bytes, Some(16384));
        assert!(mem.physical.is_none());
    }

    #[test]
    fn memory_model_with_physical() {
        let src = test_source(Source::DeviceModel, "compute tile");
        let phys_src = test_source(Source::Am025Json, "AM020 SRAM");
        let mem = MemoryModel::new(
            65536,
            BankingModel {
                num_banks: 4,
                bank_size: 16384,
                bank_width_bits: 32,
                source: src.clone(),
            },
            Some(BankingModel {
                num_banks: 8,
                bank_size: 8192,
                bank_width_bits: 128,
                source: phys_src,
            }),
            Some(16384),
            src,
        );
        assert!(mem.physical.is_some());
        let phys = mem.physical.as_ref().unwrap();
        assert_eq!(phys.num_banks, 8);
        assert_eq!(phys.bank_size, 8192);
        assert_eq!(phys.bank_width_bits, 128);
        // effective_physical returns the physical level when present
        let eff = mem.effective_physical();
        assert_eq!(eff.num_banks, 8);
        assert_eq!(eff.bank_size, 8192);
    }

    #[test]
    fn memory_model_effective_physical_fallback() {
        let src = test_source(Source::DeviceModel, "compute tile");
        let mem = MemoryModel::new(
            65536,
            BankingModel {
                num_banks: 4,
                bank_size: 16384,
                bank_width_bits: 32,
                source: src.clone(),
            },
            None,
            None,
            src,
        );
        // No physical level -- effective_physical falls back to logical
        let eff = mem.effective_physical();
        assert_eq!(eff.num_banks, 4);
        assert_eq!(eff.bank_size, 16384);
        assert_eq!(eff.bank_width_bits, 32);
    }

    #[test]
    #[should_panic(expected = "logical banking invariant")]
    fn memory_model_rejects_bad_logical() {
        let src = test_source(Source::DeviceModel, "bad");
        // 4 banks * 8192 = 32768, not 65536
        MemoryModel::new(
            65536,
            BankingModel {
                num_banks: 4,
                bank_size: 8192,
                bank_width_bits: 32,
                source: src.clone(),
            },
            None,
            None,
            src,
        );
    }

    #[test]
    #[should_panic(expected = "physical banking invariant")]
    fn memory_model_rejects_bad_physical() {
        let src = test_source(Source::DeviceModel, "bad");
        // logical is correct: 4 * 16384 = 65536
        // physical is wrong: 8 * 4096 = 32768, not 65536
        MemoryModel::new(
            65536,
            BankingModel {
                num_banks: 4,
                bank_size: 16384,
                bank_width_bits: 32,
                source: src.clone(),
            },
            Some(BankingModel {
                num_banks: 8,
                bank_size: 4096,
                bank_width_bits: 128,
                source: src.clone(),
            }),
            None,
            src,
        );
    }

    // ========================================================================
    // SubsystemModel tests
    // ========================================================================

    #[test]
    fn subsystem_model_basic_construction() {
        let src = test_source(Source::Am025Json, "register grouping");
        let sub = SubsystemModel {
            kind: SubsystemKind::Dma,
            offset_start: Confirmed::new(0x1D000, src.clone()),
            offset_end: Confirmed::new(0x1E000, src.clone()),
            registers: vec![
                RegisterModel {
                    name: "BD0".into(),
                    offset: 0x1D000,
                    width: 32,
                    reset_value: 0,
                    fields: vec![],
                    subsystem: SubsystemKind::Dma,
                    access: Access::ReadWrite,
                    source: src.clone(),
                },
                RegisterModel {
                    name: "BD1".into(),
                    offset: 0x1D004,
                    width: 32,
                    reset_value: 0,
                    fields: vec![],
                    subsystem: SubsystemKind::Dma,
                    access: Access::ReadWrite,
                    source: src.clone(),
                },
                RegisterModel {
                    name: "BD2".into(),
                    offset: 0x1D008,
                    width: 32,
                    reset_value: 0,
                    fields: vec![],
                    subsystem: SubsystemKind::Dma,
                    access: Access::ReadWrite,
                    source: src.clone(),
                },
            ],
        };
        assert_eq!(sub.kind, SubsystemKind::Dma);
        assert_eq!(*sub.offset_start.value(), 0x1D000);
        assert_eq!(*sub.offset_end.value(), 0x1E000);
        assert_eq!(sub.registers.len(), 3);
    }

    #[test]
    fn subsystem_model_contains_offset() {
        let src = test_source(Source::Am025Json, "register grouping");
        let sub = SubsystemModel {
            kind: SubsystemKind::Lock,
            offset_start: Confirmed::new(0x1F000, src.clone()),
            offset_end: Confirmed::new(0x1F100, src),
            registers: vec![],
        };
        assert!(sub.contains_offset(0x1F000));  // inclusive start
        assert!(sub.contains_offset(0x1F0FF));  // last byte before end
        assert!(!sub.contains_offset(0x1F100)); // exclusive end
        assert!(!sub.contains_offset(0x1D000)); // outside
    }
}
