//! CDO (Configuration Data Object) parser.
//!
//! See docs/superpowers/specs/2026-04-23-subsys8-parser-design.md for the
//! two-layer design: framing (byte-level), syntax (CdoRaw typed
//! commands), semantics (CdoRaw -> DeviceOp in Half 2; pass-through
//! in Half 1).

pub mod framing;
pub mod syntax;
pub mod semantics;

pub use framing::{find_cdo_offset, CdoVersion, RawCdoHeader, CDO_MAGIC_CDO, CDO_MAGIC_XLNX, CDO_HEADER_SIZE};
pub use syntax::{Cdo, CdoOpcode, CdoRaw};
