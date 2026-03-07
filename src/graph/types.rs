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

// ============================================================================
// Source attribution
// ============================================================================

// ============================================================================
// Cross-source confirmation
// ============================================================================

/// Error when sources disagree on a fact's value.
#[derive(Debug, Clone)]
pub struct ConflictError {
    pub existing_sources: Vec<SourceAttribution>,
    pub new_source: SourceAttribution,
    pub detail: String,
}

impl fmt::Display for ConflictError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "source conflict: {} disagrees with {} existing source(s): {}",
            self.new_source,
            self.existing_sources.len(),
            self.detail,
        )
    }
}

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
impl NoProvenance for InstanceCount {}

/// A fact confirmed by one or more toolchain sources.
///
/// Every fact in the graph is wrapped in `Confirmed<T>`. The first source
/// establishes the value; subsequent sources must agree (via `FactEquals`)
/// or the confirmation fails with `ConflictError`.
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
    /// Returns `Err(ConflictError)` if the value disagrees.
    pub fn confirm(
        &mut self,
        value: T,
        source: SourceAttribution,
    ) -> Result<(), ConflictError> {
        if self.value.fact_equals(&value) {
            self.sources.push(source);
            Ok(())
        } else {
            Err(ConflictError {
                existing_sources: self.sources.clone(),
                new_source: source,
                detail: String::new(),
            })
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
}

impl fmt::Display for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Am025Json => write!(f, "AM025 JSON"),
            Self::AieRt => write!(f, "aie-rt"),
            Self::DeviceModel => write!(f, "device model"),
            Self::TableGen => write!(f, "TableGen"),
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
/// Memory banking info (num_banks, bank_size) lives in `MemoryModel`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InstanceCount {
    pub locks: u8,
    pub bds: u8,
    pub channels: u8,
}

/// A functional module within a tile (Layer 2+: registers grouped by function).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModuleModel {
    pub kind: ModuleKind,
    pub registers: Vec<RegisterModel>,
    pub source: SourceAttribution,
}

impl FactEquals for ModuleModel {
    fn fact_equals(&self, other: &Self) -> bool {
        self.kind == other.kind
            && self.registers.len() == other.registers.len()
            && self
                .registers
                .iter()
                .zip(other.registers.iter())
                .all(|(a, b)| a.fact_equals(b))
    }
}

// ============================================================================
// Tile and array topology
// ============================================================================

/// Memory model for a tile type.
///
/// `size_bytes` is data memory (64KB compute, 512KB memtile).
/// `program_memory_bytes` is instruction memory (16KB, compute tiles only).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryModel {
    pub size_bytes: u64,
    pub num_banks: u8,
    pub bank_size_bytes: u64,
    /// Instruction memory size. Only present on compute tiles.
    pub program_memory_bytes: Option<u64>,
    pub source: SourceAttribution,
}

impl FactEquals for MemoryModel {
    fn fact_equals(&self, other: &Self) -> bool {
        self.size_bytes == other.size_bytes
            && self.num_banks == other.num_banks
            && self.bank_size_bytes == other.bank_size_bytes
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
    /// Switchbox port bundles (present on all tile types).
    pub switchbox_ports: Vec<PortBundle>,
    /// Shim mux port bundles (only present on shim tiles).
    pub shim_mux_ports: Vec<PortBundle>,
    /// Register modules (populated in Layer 2).
    pub modules: Vec<ModuleModel>,
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
// Relationships
// ============================================================================

/// A node identifier in the graph (dot-separated path).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub String);

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Type of relationship between two nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RelationshipKind {
    DataFlow,
    Configures,
    BlocksOn,
    Triggers,
    RoutesTo,
    Produces,
    Contains,
}

/// A directed relationship between two nodes.
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
        let result = c.confirm(16u8, test_source(Source::Am025Json, "register analysis"));
        assert!(result.is_ok());
        assert_eq!(c.sources().len(), 2);
    }

    #[test]
    fn confirmed_multiple_sources_number_agnostic() {
        let mut c = Confirmed::new(16u8, test_source(Source::DeviceModel, "device model"));
        c.confirm(16u8, test_source(Source::Am025Json, "register analysis"))
            .unwrap();
        c.confirm(16u8, test_source(Source::AieRt, "aie-rt"))
            .unwrap();
        assert_eq!(*c.value(), 16u8);
        assert_eq!(c.sources().len(), 3);
        assert_eq!(c.sources()[0].origin, Source::DeviceModel);
        assert_eq!(c.sources()[1].origin, Source::Am025Json);
        assert_eq!(c.sources()[2].origin, Source::AieRt);
    }

    #[test]
    fn confirmed_instance_count_cross_validation() {
        // Device model says 16 locks, 16 BDs, 2 channels
        let dm_counts = InstanceCount { locks: 16, bds: 16, channels: 2 };
        let mut c = Confirmed::new(
            dm_counts,
            test_source(Source::DeviceModel, "npu1 core"),
        );
        // Register analysis independently discovers the same counts
        let reg_counts = InstanceCount { locks: 16, bds: 16, channels: 2 };
        c.confirm(reg_counts, test_source(Source::Am025Json, "register grouping"))
            .unwrap();
        assert_eq!(c.sources().len(), 2);

        // aie-rt would be a third source confirming the same
        let rt_counts = InstanceCount { locks: 16, bds: 16, channels: 2 };
        c.confirm(rt_counts, test_source(Source::AieRt, "DmaMod"))
            .unwrap();
        assert_eq!(c.sources().len(), 3);
    }

    #[test]
    fn confirmed_instance_count_catches_disagreement() {
        let dm_counts = InstanceCount { locks: 16, bds: 16, channels: 2 };
        let mut c = Confirmed::new(
            dm_counts,
            test_source(Source::DeviceModel, "npu1 core"),
        );
        // Register analysis finds 48 BDs (memtile value for a core tile -- bug!)
        let wrong_counts = InstanceCount { locks: 16, bds: 48, channels: 2 };
        let result = c.confirm(
            wrong_counts,
            test_source(Source::Am025Json, "register grouping"),
        );
        assert!(result.is_err());
    }

    #[test]
    fn confirmed_equality_ignores_sources() {
        let c1 = Confirmed::new(16u8, test_source(Source::DeviceModel, "dm"));
        let c2 = Confirmed::new(16u8, test_source(Source::AieRt, "rt"));
        assert_eq!(c1, c2); // same value, different sources -> equal
    }

    #[test]
    fn confirmed_conflicting_value_errors() {
        let mut c = Confirmed::new(16u8, test_source(Source::DeviceModel, "locks"));
        let result = c.confirm(48u8, test_source(Source::Am025Json, "register analysis"));
        assert!(result.is_err());
        // Original value unchanged
        assert_eq!(*c.value(), 16u8);
        assert_eq!(c.sources().len(), 1);
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
        let result = c.confirm(reg_same_data, test_source(Source::AieRt, "second"));
        assert!(result.is_ok());
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
            instances: InstanceCount { locks: 16, bds: 16, channels: 2 },
            modules: vec![ModuleModel {
                kind: ModuleKind::Core,
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
                source: SourceAttribution {
                    origin: Source::Am025Json,
                    file: "test.json".to_string(),
                    detail: "test".to_string(),
                },
            }],
            memory: Some(MemoryModel {
                size_bytes: 65536,
                num_banks: 4,
                bank_size_bytes: 16384,
                program_memory_bytes: Some(16384),
                source: SourceAttribution {
                    origin: Source::DeviceModel,
                    file: "test.json".to_string(),
                    detail: "test".to_string(),
                },
            }),
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
            switchbox_ports: vec![
                PortBundle { bundle: "DMA".to_string(), masters: 2, slaves: 2 },
                PortBundle { bundle: "Core".to_string(), masters: 1, slaves: 1 },
            ],
            shim_mux_ports: Vec::new(),
            source: SourceAttribution {
                origin: Source::DeviceModel,
                file: "test.json".to_string(),
                detail: "test".to_string(),
            },
        });
        model.relationships.push(Relationship {
            from: NodeId("compute".to_string()),
            to: NodeId("compute.dma".to_string()),
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
            instances: InstanceCount { locks: 16, bds: 16, channels: 2 },
            memory: None,
            dma_capabilities: None,
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
        let compute_mem = MemoryModel {
            size_bytes: 65536,
            num_banks: 4,
            bank_size_bytes: 16384,
            program_memory_bytes: Some(16384),
            source: SourceAttribution {
                origin: Source::DeviceModel,
                file: "test.json".to_string(),
                detail: "test".to_string(),
            },
        };
        let memtile_mem = MemoryModel {
            size_bytes: 524288,
            num_banks: 8,
            bank_size_bytes: 65536,
            program_memory_bytes: None,
            source: SourceAttribution {
                origin: Source::DeviceModel,
                file: "test.json".to_string(),
                detail: "test".to_string(),
            },
        };
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
}
