//! Stream switch subsystem signal mapping.
//!
//! VCD hierarchy for each stream port (from aiesimulator output):
//!
//! ```text
//! tile_0_1.stream_switch.from_sSouth3.data                    (32-bit)
//! tile_0_1.stream_switch.from_sSouth3.event_idle_sSouth3      (1-bit)
//! tile_0_1.stream_switch.from_sSouth3.event_running_sSouth3   (1-bit)
//! tile_0_1.stream_switch.from_sSouth3.event_stalled_sSouth3   (1-bit)
//! tile_0_1.stream_switch.from_sSouth3.event_tlast_sSouth3     (1-bit)
//! ```
//!
//! The scope name is `stream_switch`. Under it, each port has a sub-scope
//! `from_{port_name}` (e.g., `from_sSouth3`) containing:
//! - `data` -- 32-bit data word on the port
//! - `event_idle_{port_name}` -- 1-bit idle status
//! - `event_running_{port_name}` -- 1-bit active transfer status
//! - `event_stalled_{port_name}` -- 1-bit stalled status
//! - `event_tlast_{port_name}` -- 1-bit TLAST (end of packet) signal
//!
//! **Naming note:** The event signals append the port name as a suffix
//! (e.g., `event_idle_sSouth3`, not just `event_idle`). This was observed
//! from aiesimulator VCD output and must match exactly for resolution to work.
//!
//! # Usage
//!
//! Build a mapping for a known set of port names:
//!
//! ```rust
//! use xdna_emu::vcd::stream_mapping::stream_mapping;
//! let mapping = stream_mapping(&["sSouth3", "sDMA0"]);
//! assert_eq!(mapping.scope_name(), "stream_switch");
//! ```
//!
//! Or use the per-tile-type helpers for the standard AIE2 (Phoenix) port sets:
//!
//! ```rust
//! use xdna_emu::vcd::stream_mapping::compute_stream_ports;
//! let ports = compute_stream_ports();
//! assert!(!ports.is_empty());
//! ```
//!
//! # Port name sets
//!
//! Port names are derived from the mlir-aie device model and the VCD naming
//! convention where bundle names use the "s{Bundle}{index}" format:
//! - Compute tiles: sSouth0-3, sNorth0-5, sEast0-1, sWest0-1, sDMA0-1,
//!   sCore0-1, sTrace0-1, sTileCtrl
//! - Mem tiles: sSouth0-3, sNorth0-5, sDMA0-5
//! - Shim tiles: sSouth0-3, sNorth0-5, sDMA0-1

use crate::vcd::mapping::{SubsystemMapping, TileMapping};
use crate::vcd::state_path::{PortId, StatePath, Subsystem};

// ---------------------------------------------------------------------------
// Port name set helpers
// ---------------------------------------------------------------------------

/// Standard stream port names for a compute tile (AIE2 Phoenix array tile).
///
/// Port names follow the aiesimulator VCD naming convention
/// (`s{Bundle}{index}`). The set is derived from the mlir-aie device model
/// for the `npu1` compute tile type.
///
/// Source: mlir-aie AIETargetModel + aiesimulator VCD output convention.
pub fn compute_stream_ports() -> Vec<String> {
    let mut ports = Vec::new();
    // South: 4 ports (sSouth0 - sSouth3)
    for i in 0..4u8 {
        ports.push(format!("sSouth{}", i));
    }
    // North: 6 ports (sNorth0 - sNorth5)
    for i in 0..6u8 {
        ports.push(format!("sNorth{}", i));
    }
    // East: 2 ports
    for i in 0..2u8 {
        ports.push(format!("sEast{}", i));
    }
    // West: 2 ports
    for i in 0..2u8 {
        ports.push(format!("sWest{}", i));
    }
    // DMA: 2 ports
    for i in 0..2u8 {
        ports.push(format!("sDMA{}", i));
    }
    // Core: 2 ports
    for i in 0..2u8 {
        ports.push(format!("sCore{}", i));
    }
    // Trace: 2 ports
    for i in 0..2u8 {
        ports.push(format!("sTrace{}", i));
    }
    // Tile control: 1 port
    ports.push("sTileCtrl".to_string());
    ports
}

/// Standard stream port names for a mem tile (AIE2 Phoenix mem tile).
///
/// Source: mlir-aie AIETargetModel for `npu1` mem tile type.
pub fn memtile_stream_ports() -> Vec<String> {
    let mut ports = Vec::new();
    // South: 4 ports
    for i in 0..4u8 {
        ports.push(format!("sSouth{}", i));
    }
    // North: 6 ports
    for i in 0..6u8 {
        ports.push(format!("sNorth{}", i));
    }
    // DMA: 6 ports
    for i in 0..6u8 {
        ports.push(format!("sDMA{}", i));
    }
    ports
}

/// Standard stream port names for a shim tile (AIE2 Phoenix shim row).
///
/// Source: mlir-aie AIETargetModel for `npu1` shim tile type.
pub fn shim_stream_ports() -> Vec<String> {
    let mut ports = Vec::new();
    // South: 4 ports
    for i in 0..4u8 {
        ports.push(format!("sSouth{}", i));
    }
    // North: 6 ports
    for i in 0..6u8 {
        ports.push(format!("sNorth{}", i));
    }
    // DMA: 2 ports
    for i in 0..2u8 {
        ports.push(format!("sDMA{}", i));
    }
    ports
}


// ---------------------------------------------------------------------------
// StreamPortMapping -- custom mapping for stream ports
// ---------------------------------------------------------------------------

/// Mapping entry for a single stream port.
struct PortEntry {
    /// VCD scope name for this port: `from_{port_name}`.
    scope: String,
    /// PortId for this port.
    port_id: PortId,
}

/// Custom [`SubsystemMapping`]-compatible type for stream switch ports.
///
/// The standard [`SubsystemMapping`] handles integer-indexed groups
/// (`prefix + number`). Stream ports use string-keyed scopes like
/// `from_sSouth3` (prefix `from_` + port name, not a number). This type
/// handles resolution manually.
///
/// Exposed via the [`stream_mapping`] constructor which wraps this into a
/// pattern that the mapping tree builder can use.
pub struct StreamPortMapping {
    ports: Vec<PortEntry>,
}

impl StreamPortMapping {
    fn new(port_names: &[&str]) -> Self {
        StreamPortMapping {
            ports: port_names
                .iter()
                .map(|name| PortEntry {
                    scope: format!("from_{}", name),
                    port_id: PortId::named(name),
                })
                .collect(),
        }
    }

    /// Attempt to resolve VCD segments to a [`StatePath`].
    ///
    /// Handles two formats:
    ///
    /// **Nested (2 segments):** `["from_{port}", "{signal}"]`
    /// - `from_sSouth3` + `data` -> [`StatePath::StreamPortData`]
    /// - `from_sSouth3` + `event_idle_sSouth3` -> [`StatePath::StreamPortIdle`]
    ///
    /// **Flat (1 segment):** `["event_idle_{port}"]` or `["from_{port}"]`
    /// - In real aiesimulator VCDs, event signals are flat at the
    ///   `stream_switch` level: `event_idle_sSouth0`, not nested
    ///   under `from_sSouth0`. Data signals remain nested.
    pub fn resolve(&self, segments: &[&str], col: u8, row: u8) -> Option<StatePath> {
        match segments.len() {
            1 => self.resolve_flat(segments[0], col, row),
            2 => self.resolve_nested(segments[0], segments[1], col, row),
            _ => None,
        }
    }

    /// Resolve a flat signal name like `"event_idle_sSouth3"`.
    fn resolve_flat(&self, signal: &str, col: u8, row: u8) -> Option<StatePath> {
        for entry in &self.ports {
            let port_name = entry.port_id.name();
            if signal == format!("event_idle_{}", port_name) {
                return Some(StatePath::StreamPortIdle {
                    col, row, port: PortId::named(port_name),
                });
            }
            if signal == format!("event_running_{}", port_name) {
                return Some(StatePath::StreamPortRunning {
                    col, row, port: PortId::named(port_name),
                });
            }
            if signal == format!("event_stalled_{}", port_name) {
                return Some(StatePath::StreamPortStalled {
                    col, row, port: PortId::named(port_name),
                });
            }
            if signal == format!("event_tlast_{}", port_name) {
                return Some(StatePath::StreamPortTlast {
                    col, row, port: PortId::named(port_name),
                });
            }
        }
        None
    }

    /// Resolve a nested signal `["from_{port}", "{leaf}"]`.
    fn resolve_nested(&self, scope: &str, signal: &str, col: u8, row: u8) -> Option<StatePath> {
        let port_id = self.ports.iter().find(|e| e.scope == scope).map(|e| &e.port_id)?;
        let port_name = port_id.name();

        match signal {
            "data" => Some(StatePath::StreamPortData {
                col, row, port: PortId::named(port_name),
            }),
            _ if signal == format!("event_idle_{}", port_name) => {
                Some(StatePath::StreamPortIdle {
                    col, row, port: PortId::named(port_name),
                })
            }
            _ if signal == format!("event_running_{}", port_name) => {
                Some(StatePath::StreamPortRunning {
                    col, row, port: PortId::named(port_name),
                })
            }
            _ if signal == format!("event_stalled_{}", port_name) => {
                Some(StatePath::StreamPortStalled {
                    col, row, port: PortId::named(port_name),
                })
            }
            _ if signal == format!("event_tlast_{}", port_name) => {
                Some(StatePath::StreamPortTlast {
                    col, row, port: PortId::named(port_name),
                })
            }
            _ => None,
        }
    }

    /// Enumerate all [`StatePath`] values this mapping can produce for a tile.
    pub fn enumerate(&self, col: u8, row: u8) -> Vec<StatePath> {
        let mut paths = Vec::new();
        for entry in &self.ports {
            let port = entry.port_id.clone();
            paths.push(StatePath::StreamPortData { col, row, port: port.clone() });
            paths.push(StatePath::StreamPortIdle { col, row, port: port.clone() });
            paths.push(StatePath::StreamPortRunning { col, row, port: port.clone() });
            paths.push(StatePath::StreamPortStalled { col, row, port: port.clone() });
            paths.push(StatePath::StreamPortTlast { col, row, port });
        }
        paths
    }

    /// The VCD scope name for the stream switch subsystem.
    pub fn scope_name(&self) -> &str {
        "stream_switch"
    }
}

impl TileMapping for StreamPortMapping {
    fn scope_name(&self) -> &str {
        StreamPortMapping::scope_name(self)
    }

    fn resolve(&self, segments: &[&str], col: u8, row: u8) -> Option<StatePath> {
        StreamPortMapping::resolve(self, segments, col, row)
    }

    fn enumerate(&self, col: u8, row: u8) -> Vec<StatePath> {
        StreamPortMapping::enumerate(self, col, row)
    }
}

// ---------------------------------------------------------------------------
// Public API -- SubsystemMapping adapter
// ---------------------------------------------------------------------------

/// Build the stream switch subsystem mapping for a tile with the given ports.
///
/// Returns a [`SubsystemMapping`] covering the VCD `stream_switch` scope. Each
/// port in `port_names` generates a `from_{port}` sub-scope with five signals:
/// `data`, `event_idle_{port}`, `event_running_{port}`, `event_stalled_{port}`,
/// and `event_tlast_{port}`.
///
/// # Signal naming (ground truth from aiesimulator VCD)
///
/// The event signals include the port name as a suffix, e.g.:
/// - `tile_0_1.stream_switch.from_sSouth3.data`
/// - `tile_0_1.stream_switch.from_sSouth3.event_idle_sSouth3`
/// - `tile_0_1.stream_switch.from_sSouth3.event_running_sSouth3`
/// - `tile_0_1.stream_switch.from_sSouth3.event_stalled_sSouth3`
/// - `tile_0_1.stream_switch.from_sSouth3.event_tlast_sSouth3`
///
/// # Note on SubsystemMapping
///
/// The standard [`SubsystemMapping`] only handles integer-indexed nested
/// groups. Stream ports use string-keyed scopes (`from_sSouth3`, not
/// `from_0`). This function wraps a [`StreamPortMapping`] into a
/// [`SubsystemMapping`] using a custom-resolution approach: fixed signals
/// are registered with full two-level names that encode the port scope.
///
/// For now, this function returns a [`StreamPortMapping`] wrapped in a
/// thin adapter. The mapping tree's `subsystem()` builder method accepts
/// [`SubsystemMapping`] -- so we implement resolution inside the returned
/// type. The `mapping_tree()` API is extended to also accept
/// [`StreamPortMapping`] via a separate registration path.
///
/// Since the mapping tree currently only accepts [`SubsystemMapping`], and
/// extending it is outside this task's scope, stream_mapping returns a
/// dedicated [`StreamPortMapping`] for direct use in tests and in Task 8's
/// tree construction.
pub fn stream_mapping(port_names: &[&str]) -> StreamPortMapping {
    StreamPortMapping::new(port_names)
}

/// Build the stream switch mapping for a compute tile using the standard
/// AIE2 Phoenix port set.
pub fn compute_stream_mapping() -> StreamPortMapping {
    let ports = compute_stream_ports();
    let port_refs: Vec<&str> = ports.iter().map(|s| s.as_str()).collect();
    StreamPortMapping::new(&port_refs)
}

/// Build the stream switch mapping for a mem tile using the standard
/// AIE2 Phoenix port set.
pub fn memtile_stream_mapping() -> StreamPortMapping {
    let ports = memtile_stream_ports();
    let port_refs: Vec<&str> = ports.iter().map(|s| s.as_str()).collect();
    StreamPortMapping::new(&port_refs)
}

/// Build the stream switch mapping for a shim tile using the standard
/// AIE2 Phoenix port set.
pub fn shim_stream_mapping() -> StreamPortMapping {
    let ports = shim_stream_ports();
    let port_refs: Vec<&str> = ports.iter().map(|s| s.as_str()).collect();
    StreamPortMapping::new(&port_refs)
}

// ---------------------------------------------------------------------------
// MappingTree integration -- SubsystemMapping adapter
// ---------------------------------------------------------------------------

/// Convert a [`StreamPortMapping`] into a [`SubsystemMapping`] for use with
/// the standard mapping tree builder.
///
/// Since [`SubsystemMapping`] supports integer-indexed nested groups but
/// stream ports are string-keyed, this adapter works by registering each
/// port's signals as flat signals with composite names. For each port
/// `sSouth3`, it registers:
/// - `"from_sSouth3\x00data"` (using a null byte as scope separator for
///   internal encoding)
///
/// This approach is internal to this adapter and is transparent to the
/// mapping tree, which splits segments at the tree level before passing
/// them to `SubsystemMapping::resolve()`.
///
/// The cleaner long-term solution is to extend `SubsystemMapping` with a
/// `string_keyed_group()` builder method (Task 8). For now, this adapter
/// lets the stream mapping integrate with the existing tree.
///
/// The implementation uses [`SubsystemMapping`]'s `nested_group` with one
/// twist: each port creates a group with the port scope as the group name
/// (using count=1 and index ignored), and the child signals carry the
/// port-specific event suffixes.
///
/// Because `NestedSignalFactory` receives `(col, row, group_idx)` but our
/// group_idx is always 0 (count=1) and port identity comes from the group
/// scope name, we embed the port index (position in the port list) as the
/// group prefix. The group prefix is `"from_portname"` with count=1.
///
/// Actually: the `nested_group` API matches `{prefix}{idx}` where idx is
/// a decimal number. "from_sSouth3" contains no number suffix, so it would
/// never match. This approach doesn't work.
///
/// The real solution: extend `SubsystemMapping` with a `resolve_hook` trait
/// method, OR represent stream ports as flat signals with two-part encoded
/// names, OR leave `StreamPortMapping` as a standalone type and extend the
/// mapping tree to accept it directly (Task 8).
///
/// For Task 7, we leave `StreamPortMapping` as a standalone type. It is
/// fully functional for direct use and testing. Task 8 will wire it into
/// the tree.
pub fn stream_mapping_as_subsystem(port_names: &[&str]) -> SubsystemMapping {
    // This is a temporary placeholder that creates an empty SubsystemMapping
    // with the correct scope name. The actual stream port resolution is handled
    // by StreamPortMapping in a separate code path. Task 8 will integrate both.
    //
    // This function exists so that callers who need a SubsystemMapping (e.g.,
    // for building trees before stream support is fully wired) get a correctly-
    // named but empty mapping. The coverage audit will show stream signals as
    // unmapped, which is the correct signal for Task 8.
    let _ = port_names;
    SubsystemMapping::new("stream_switch", Subsystem::Stream)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vcd::state_path::{PortId, StatePath};

    // -- StreamPortMapping resolution tests --

    #[test]
    fn stream_resolves_port_data() {
        let mapping = stream_mapping(&["sSouth3", "sDMA0"]);
        let result = mapping.resolve(&["from_sSouth3", "data"], 0, 1);
        assert_eq!(
            result,
            Some(StatePath::StreamPortData {
                col: 0,
                row: 1,
                port: PortId::named("sSouth3"),
            })
        );
    }

    #[test]
    fn stream_resolves_port_idle() {
        // Event signals carry the port name as a suffix in aiesimulator VCD output.
        let mapping = stream_mapping(&["sSouth3"]);
        let result = mapping.resolve(&["from_sSouth3", "event_idle_sSouth3"], 0, 1);
        assert_eq!(
            result,
            Some(StatePath::StreamPortIdle {
                col: 0,
                row: 1,
                port: PortId::named("sSouth3"),
            })
        );
    }

    #[test]
    fn stream_resolves_port_running() {
        let mapping = stream_mapping(&["sSouth3"]);
        let result = mapping.resolve(&["from_sSouth3", "event_running_sSouth3"], 0, 1);
        assert_eq!(
            result,
            Some(StatePath::StreamPortRunning {
                col: 0,
                row: 1,
                port: PortId::named("sSouth3"),
            })
        );
    }

    #[test]
    fn stream_resolves_port_stalled() {
        let mapping = stream_mapping(&["sSouth3"]);
        let result = mapping.resolve(&["from_sSouth3", "event_stalled_sSouth3"], 2, 3);
        assert_eq!(
            result,
            Some(StatePath::StreamPortStalled {
                col: 2,
                row: 3,
                port: PortId::named("sSouth3"),
            })
        );
    }

    #[test]
    fn stream_resolves_port_tlast() {
        let mapping = stream_mapping(&["sSouth3"]);
        let result = mapping.resolve(&["from_sSouth3", "event_tlast_sSouth3"], 1, 2);
        assert_eq!(
            result,
            Some(StatePath::StreamPortTlast {
                col: 1,
                row: 2,
                port: PortId::named("sSouth3"),
            })
        );
    }

    #[test]
    fn stream_resolves_second_port() {
        // Verifies that having multiple ports in the mapping doesn't cause cross-port confusion.
        let mapping = stream_mapping(&["sSouth3", "sDMA0"]);
        let result = mapping.resolve(&["from_sDMA0", "data"], 0, 1);
        assert_eq!(
            result,
            Some(StatePath::StreamPortData {
                col: 0,
                row: 1,
                port: PortId::named("sDMA0"),
            })
        );
    }

    #[test]
    fn stream_resolves_dma_port_idle() {
        let mapping = stream_mapping(&["sDMA0"]);
        let result = mapping.resolve(&["from_sDMA0", "event_idle_sDMA0"], 0, 1);
        assert_eq!(
            result,
            Some(StatePath::StreamPortIdle {
                col: 0,
                row: 1,
                port: PortId::named("sDMA0"),
            })
        );
    }

    #[test]
    fn stream_rejects_wrong_port_scope() {
        let mapping = stream_mapping(&["sSouth3"]);
        // "from_sNorth0" is not in the mapping.
        assert_eq!(mapping.resolve(&["from_sNorth0", "data"], 0, 1), None);
    }

    #[test]
    fn stream_rejects_unknown_signal() {
        let mapping = stream_mapping(&["sSouth3"]);
        assert_eq!(
            mapping.resolve(&["from_sSouth3", "nonexistent_signal"], 0, 1),
            None
        );
    }

    #[test]
    fn stream_rejects_event_signal_with_wrong_port_suffix() {
        // "event_idle_sNorth0" is not valid for the "from_sSouth3" scope.
        let mapping = stream_mapping(&["sSouth3"]);
        assert_eq!(
            mapping.resolve(&["from_sSouth3", "event_idle_sNorth0"], 0, 1),
            None
        );
    }

    #[test]
    fn stream_rejects_wrong_segment_count() {
        let mapping = stream_mapping(&["sSouth3"]);
        // Too few segments (need exactly 2).
        assert_eq!(mapping.resolve(&["from_sSouth3"], 0, 1), None);
        // Too many segments.
        assert_eq!(mapping.resolve(&["from_sSouth3", "data", "extra"], 0, 1), None);
    }

    #[test]
    fn stream_tile_coordinates_propagated() {
        let mapping = stream_mapping(&["sSouth3"]);
        let a = mapping.resolve(&["from_sSouth3", "data"], 0, 1).unwrap();
        let b = mapping.resolve(&["from_sSouth3", "data"], 3, 5).unwrap();
        assert_ne!(a, b);
        assert_eq!(a, StatePath::StreamPortData {
            col: 0,
            row: 1,
            port: PortId::named("sSouth3"),
        });
        assert_eq!(b, StatePath::StreamPortData {
            col: 3,
            row: 5,
            port: PortId::named("sSouth3"),
        });
    }

    #[test]
    fn stream_scope_name_is_stream_switch() {
        let mapping = stream_mapping(&["sSouth3"]);
        assert_eq!(mapping.scope_name(), "stream_switch");
    }

    // -- Enumeration tests --

    #[test]
    fn stream_enumerates_five_signals_per_port() {
        let mapping = stream_mapping(&["sSouth3"]);
        let paths = mapping.enumerate(0, 1);
        // 1 port x 5 signals = 5
        assert_eq!(paths.len(), 5);
    }

    #[test]
    fn stream_enumerates_all_ports() {
        let mapping = stream_mapping(&["sSouth3", "sDMA0", "sNorth0"]);
        let paths = mapping.enumerate(0, 1);
        // 3 ports x 5 signals = 15
        assert_eq!(paths.len(), 15);
    }

    #[test]
    fn stream_enumerate_contains_all_signal_types() {
        let mapping = stream_mapping(&["sSouth3"]);
        let paths = mapping.enumerate(0, 1);
        let port = PortId::named("sSouth3");
        assert!(paths.contains(&StatePath::StreamPortData { col: 0, row: 1, port: port.clone() }));
        assert!(paths.contains(&StatePath::StreamPortIdle { col: 0, row: 1, port: port.clone() }));
        assert!(paths.contains(&StatePath::StreamPortRunning { col: 0, row: 1, port: port.clone() }));
        assert!(paths.contains(&StatePath::StreamPortStalled { col: 0, row: 1, port: port.clone() }));
        assert!(paths.contains(&StatePath::StreamPortTlast { col: 0, row: 1, port }));
    }

    #[test]
    fn stream_enumerate_tile_coordinates_propagated() {
        let mapping = stream_mapping(&["sSouth3"]);
        let paths = mapping.enumerate(2, 4);
        assert!(paths.iter().all(|p| p.tile() == (2, 4)));
    }

    // -- Port name set tests --

    #[test]
    fn compute_ports_count() {
        // 4 South + 6 North + 2 East + 2 West + 2 DMA + 2 Core + 2 Trace + 1 TileCtrl = 21
        let ports = compute_stream_ports();
        assert_eq!(ports.len(), 21);
    }

    #[test]
    fn compute_ports_contain_expected_names() {
        let ports = compute_stream_ports();
        assert!(ports.contains(&"sSouth0".to_string()));
        assert!(ports.contains(&"sSouth3".to_string()));
        assert!(ports.contains(&"sNorth0".to_string()));
        assert!(ports.contains(&"sNorth5".to_string()));
        assert!(ports.contains(&"sDMA0".to_string()));
        assert!(ports.contains(&"sDMA1".to_string()));
        assert!(ports.contains(&"sCore0".to_string()));
        assert!(ports.contains(&"sTrace0".to_string()));
        assert!(ports.contains(&"sTileCtrl".to_string()));
    }

    #[test]
    fn memtile_ports_count() {
        // 4 South + 6 North + 6 DMA = 16
        let ports = memtile_stream_ports();
        assert_eq!(ports.len(), 16);
    }

    #[test]
    fn shim_ports_count() {
        // 4 South + 6 North + 2 DMA = 12
        let ports = shim_stream_ports();
        assert_eq!(ports.len(), 12);
    }

    #[test]
    fn compute_stream_mapping_enumerates_all_ports() {
        let mapping = compute_stream_mapping();
        let paths = mapping.enumerate(0, 2);
        // 21 ports x 5 signals = 105
        assert_eq!(paths.len(), 21 * 5);
    }

    #[test]
    fn memtile_stream_mapping_enumerates_all_ports() {
        let mapping = memtile_stream_mapping();
        let paths = mapping.enumerate(0, 1);
        // 16 ports x 5 signals = 80
        assert_eq!(paths.len(), 16 * 5);
    }

    #[test]
    fn shim_stream_mapping_enumerates_all_ports() {
        let mapping = shim_stream_mapping();
        let paths = mapping.enumerate(0, 0);
        // 12 ports x 5 signals = 60
        assert_eq!(paths.len(), 12 * 5);
    }
}
