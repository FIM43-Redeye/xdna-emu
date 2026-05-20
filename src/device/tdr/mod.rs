//! Tier C TDR (Timeout Detection & Recovery) / context-restart support.
//!
//! Exposes the per-cycle classifier that decides whether the in-flight
//! submission is progressing, completing naturally, exhausting a satisfiable
//! poll, or wedged. The actual TDR algorithm (periodic timer, two-tick stuck
//! check, recovery chain) is a driver-side concern; this module exposes the
//! signals a driver TDRs on. See:
//! - `docs/superpowers/specs/2026-05-19-interrupt-tier-c-tdr-design.md`
//! - `~/npu-work/xdna-driver/src/driver/amdxdna/aie2_tdr.c` (the driver-side
//!   algorithm this lets a driver consumer drive)

pub mod detector;

pub use detector::{QuiescenceDetector, QuiescenceStatus, StallDetector, StallStatus, TdrDiagnosis};
